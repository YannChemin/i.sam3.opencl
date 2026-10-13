/****************************************************************************
 *
 * MODULE:       i.sam3.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      C port of the CLIP byte-pair encoding tokenizer used by the
 *               SAM 3 text encoder (sam3.model.tokenizer_ve).
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <zlib.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "tokenizer.h"

#define VOCAB_FILE "bpe_simple_vocab_16e6.txt.gz"
#define N_MERGES (49152 - 256 - 2) /* Merges used by SimpleTokenizer. */
#define VOCAB_SIZE (512 + N_MERGES + 2)
#define TABLE_SIZE (1 << 18)

/* Open-addressing string table mapping keys to integers. */
struct table {
    char **keys;
    int *vals;
};

static uint32_t hash(const char *s)
{
    uint32_t h = 2166136261u;

    while (*s)
        h = (h ^ (unsigned char)*s++) * 16777619u;
    return h;
}

static void table_put(struct table *t, const char *key, int val)
{
    uint32_t i = hash(key) & (TABLE_SIZE - 1);

    while (t->keys[i]) {
        if (strcmp(t->keys[i], key) == 0) {
            t->vals[i] = val; /* The last one wins, as in dict(zip()). */
            return;
        }
        i = (i + 1) & (TABLE_SIZE - 1);
    }
    t->keys[i] = G_store(key);
    t->vals[i] = val;
}

static int table_get(const struct table *t, const char *key)
{
    uint32_t i = hash(key) & (TABLE_SIZE - 1);

    while (t->keys[i]) {
        if (strcmp(t->keys[i], key) == 0)
            return t->vals[i];
        i = (i + 1) & (TABLE_SIZE - 1);
    }
    return -1;
}

struct bpe_tokenizer {
    struct table vocab;     /* Symbol (UTF-8) -> token id. */
    struct table ranks;     /* "first second" -> merge rank. */
    char byte_char[256][3]; /* bytes_to_unicode as UTF-8 strings. */
};

/* bytes_to_unicode(): printable bytes map to themselves as code points,
 * the others to 256 + n, in byte order. */
static void bytes_to_unicode(int cp[256])
{
    int b, n = 0;

    for (b = 0; b < 256; b++) {
        if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
            (b >= 0xAE && b <= 0xFF))
            cp[b] = b;
        else
            cp[b] = 256 + n++;
    }
}

static void utf8(int cp, char *out)
{
    if (cp < 0x80) {
        out[0] = (char)cp;
        out[1] = '\0';
    }
    else {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = '\0';
    }
}

static gzFile open_vocab(char **path)
{
    const char *env = getenv("I_SAM3_OPENCL_BPE");
    const char *addon = getenv("GRASS_ADDON_BASE");
    char *cand[3] = {NULL, NULL, NULL};
    gzFile fp = NULL;
    int i;

    if (env && *env)
        cand[0] = G_store(env);
    if (addon && *addon)
        G_asprintf(&cand[1], "%s/etc/i.sam3.opencl/%s", addon, VOCAB_FILE);
    G_asprintf(&cand[2], "%s/etc/i.sam3.opencl/%s", G_gisbase(), VOCAB_FILE);
    for (i = 0; i < 3 && !fp; i++)
        if (cand[i] && (fp = gzopen(cand[i], "rb")))
            *path = G_store(cand[i]);
    if (!fp)
        G_fatal_error(_("Text tokenizer vocabulary %s not found (searched "
                        "$I_SAM3_OPENCL_BPE, <%s> and <%s>)"),
                      VOCAB_FILE, cand[1] ? cand[1] : "$GRASS_ADDON_BASE/etc",
                      cand[2]);
    for (i = 0; i < 3; i++)
        G_free(cand[i]);
    return fp;
}

struct bpe_tokenizer *tokenizer_load(void)
{
    struct bpe_tokenizer *tk = G_calloc(1, sizeof(*tk));
    char line[1024], sym[2048], *path = NULL;
    int cp[256], b, id = 0, n = 0;
    gzFile fp;

    tk->vocab.keys = G_calloc(TABLE_SIZE, sizeof(char *));
    tk->vocab.vals = G_calloc(TABLE_SIZE, sizeof(int));
    tk->ranks.keys = G_calloc(TABLE_SIZE, sizeof(char *));
    tk->ranks.vals = G_calloc(TABLE_SIZE, sizeof(int));

    /* Vocabulary: the 256 byte symbols in bytes_to_unicode order (bytes
     * that map to themselves first), the same with </w>, the merges, and
     * the start and end tokens. */
    bytes_to_unicode(cp);
    {
        int order[256], k = 0;

        for (b = 0; b < 256; b++)
            if (cp[b] < 256)
                order[k++] = b;
        for (b = 0; b < 256; b++)
            if (cp[b] >= 256)
                order[k++] = b;
        for (b = 0; b < 256; b++)
            utf8(cp[b], tk->byte_char[b]);
        for (k = 0; k < 256; k++)
            table_put(&tk->vocab, tk->byte_char[order[k]], id++);
        for (k = 0; k < 256; k++) {
            snprintf(sym, sizeof(sym), "%s</w>", tk->byte_char[order[k]]);
            table_put(&tk->vocab, sym, id++);
        }
    }

    fp = open_vocab(&path);
    G_verbose_message(_("Text tokenizer vocabulary <%s>"), path);
    G_free(path);
    if (!gzgets(fp, line, sizeof(line)))
        G_fatal_error(_("Empty tokenizer vocabulary"));
    while (n < N_MERGES && gzgets(fp, line, sizeof(line))) {
        char *sp;

        line[strcspn(line, "\r\n")] = '\0';
        sp = strchr(line, ' ');
        if (!sp)
            G_fatal_error(_("Malformed tokenizer merge <%s>"), line);
        table_put(&tk->ranks, line, n++);
        *sp = '\0';
        snprintf(sym, sizeof(sym), "%s%s", line, sp + 1);
        table_put(&tk->vocab, sym, id++);
    }
    gzclose(fp);
    if (n != N_MERGES)
        G_fatal_error(_("Tokenizer vocabulary has %d merges, expected %d"), n,
                      N_MERGES);
    table_put(&tk->vocab, "<start_of_text>", id++);
    table_put(&tk->vocab, "<end_of_text>", id++);
    if (id != VOCAB_SIZE)
        G_fatal_error(_("Tokenizer vocabulary has %d symbols, expected %d"), id,
                      VOCAB_SIZE);
    return tk;
}

/* Byte-pair encode one pre-token (already mapped through
 * bytes_to_unicode), appending its ids. */
static void bpe(const struct bpe_tokenizer *tk, const char *const *chars,
                int nchars, int *ids, int *nids, int max)
{
    char **w = G_malloc(nchars * sizeof(char *)), key[4096];
    int n = nchars, i, j;

    for (i = 0; i < n; i++) {
        if (i == n - 1)
            G_asprintf(&w[i], "%s</w>", chars[i]);
        else
            w[i] = G_store(chars[i]);
    }
    for (;;) {
        int best = -1, rank = 0x7fffffff;

        for (i = 0; i + 1 < n; i++) {
            int r;

            snprintf(key, sizeof(key), "%s %s", w[i], w[i + 1]);
            r = table_get(&tk->ranks, key);
            if (r >= 0 && r < rank) {
                rank = r;
                best = i;
            }
        }
        if (best < 0)
            break;
        {
            char *first = G_store(w[best]), *second = G_store(w[best + 1]);
            int m = 0;

            for (i = 0; i < n;) {
                if (i + 1 < n && strcmp(w[i], first) == 0 &&
                    strcmp(w[i + 1], second) == 0) {
                    char *merged;

                    G_asprintf(&merged, "%s%s", first, second);
                    G_free(w[i]);
                    G_free(w[i + 1]);
                    w[m++] = merged;
                    i += 2;
                }
                else
                    w[m++] = w[i++];
            }
            n = m;
            G_free(first);
            G_free(second);
        }
        if (n == 1)
            break;
    }
    for (j = 0; j < n; j++) {
        int v = table_get(&tk->vocab, w[j]);

        if (v < 0)
            G_fatal_error(_("Token <%s> missing from the vocabulary"), w[j]);
        if (*nids < max)
            ids[(*nids)++] = v;
        G_free(w[j]);
    }
    G_free(w);
}

int tokenizer_encode(const struct bpe_tokenizer *tk, const char *text, int *ids)
{
    static const char *contractions[] = {"'s", "'t",  "'re", "'ve",
                                         "'m", "'ll", "'d"};
    /* Room for every pre-token before the truncation to TOK_CONTEXT. */
    int all[1024], n = 0, i, k;
    size_t len = strlen(text);
    char *s = G_malloc(len + 1);
    const char *chars[1024];
    size_t j = 0;

    /* whitespace_clean(basic_clean()).lower(), for ASCII text. */
    for (i = 0; text[i]; i++)
        if ((unsigned char)text[i] >= 0x80)
            G_fatal_error(_("Text prompt <%s> is not plain ASCII"), text);
    for (i = 0; text[i]; i++) {
        if (isspace((unsigned char)text[i])) {
            if (j > 0 && s[j - 1] != ' ')
                s[j++] = ' ';
        }
        else
            s[j++] = (char)tolower((unsigned char)text[i]);
    }
    while (j > 0 && s[j - 1] == ' ')
        j--;
    s[j] = '\0';

    all[n++] = table_get(&tk->vocab, "<start_of_text>");
    /* The pre-tokenizing pattern: contractions, letter runs, single
     * digits, runs of other non-space characters. */
    for (i = 0; s[i];) {
        int start = i, end = i, matched = 0;

        if (s[i] == ' ') {
            i++;
            continue;
        }
        for (k = 0; k < 7 && !matched; k++) {
            size_t cl = strlen(contractions[k]);

            if (strncmp(s + i, contractions[k], cl) == 0) {
                end = i + (int)cl;
                matched = 1;
            }
        }
        if (!matched && isalpha((unsigned char)s[i])) {
            for (end = i; isalpha((unsigned char)s[end]); end++)
                ;
            matched = 1;
        }
        if (!matched && isdigit((unsigned char)s[i])) {
            end = i + 1;
            matched = 1;
        }
        if (!matched)
            for (end = i;
                 s[end] && s[end] != ' ' && !isalnum((unsigned char)s[end]);
                 end++)
                ;
        for (k = start; k < end; k++)
            chars[k - start] = tk->byte_char[(unsigned char)s[k]];
        bpe(tk, chars, end - start, all, &n, 1023);
        i = end;
    }
    all[n++] = table_get(&tk->vocab, "<end_of_text>");
    G_free(s);

    if (n > TOK_CONTEXT) {
        n = TOK_CONTEXT;
        all[n - 1] = table_get(&tk->vocab, "<end_of_text>");
    }
    memcpy(ids, all, n * sizeof(int));
    return n;
}

void tokenizer_free(struct bpe_tokenizer *tk)
{
    int i;

    if (!tk)
        return;
    for (i = 0; i < TABLE_SIZE; i++) {
        G_free(tk->vocab.keys[i]);
        G_free(tk->ranks.keys[i]);
    }
    G_free(tk->vocab.keys);
    G_free(tk->vocab.vals);
    G_free(tk->ranks.keys);
    G_free(tk->ranks.vals);
    G_free(tk);
}
