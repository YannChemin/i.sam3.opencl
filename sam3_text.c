/****************************************************************************
 *
 * MODULE:       i.sam3.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      SAM 3 text-prompted concept detection on OpenCL: text
 *               encoder, detector neck, geometry encoder, fusion encoder,
 *               DETR-like decoder with presence token and box relative
 *               position bias, dot-product scoring and segmentation head
 *               (Sam3Image.forward_grounding as Sam3Processor runs it).
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "sam3_ops.h"
#include "sam3_text.h"
#include "tokenizer.h"

#define D SAM_DIM       /* Detector width. */
#define NH 8            /* Detector attention heads. */
#define HD (D / NH)     /* Detector head width. */
#define NQ TEXT_QUERIES /* Object queries. */
#define NQP (NQ + 1)    /* Queries and the presence token. */
#define FFN 2048        /* Detector feed-forward width. */
#define T72 (SAM_GRID * SAM_GRID)
#define T288 (16 * T72)
#define EPS 1e-5f /* nn.LayerNorm and nn.GroupNorm default. */

#define DET "detector"
#define TXT "detector.backbone.language_backbone"

/* nn.MultiheadAttention, its in_proj split into q, k and v. */
struct mha {
    cl_mem qw, qb, kw, kb, vw, vb, ow, ob;
};

/* Pre-norm self-attention, cross-attention and feed-forward layer of the
 * fusion and geometry encoders (TransformerEncoderLayer). */
struct xlayer {
    struct mha sa, ca;
    cl_mem n1w, n1b, n2w, n2b, n3w, n3b, l1w, l1b, l2w, l2b;
};

/* Post-norm decoder layer: self-attention, text and image
 * cross-attention, feed-forward (TransformerDecoderLayer). */
struct dlayer {
    struct mha sa, ct, ca;
    cl_mem n1w, n1b, n2w, n2b, n3w, n3b, ctw, ctb, l1w, l1b, l2w, l2b;
};

/* Host MLP: Linear layers with ReLU in between. */
struct hmlp {
    int nl, in[3], out[3];
    float *w[3], *b[3];
};

struct sam3_text {
    struct sam_model *m;
    cl_kernel k_soft, k_gn, k_up2;

    /* Encoded prompts: ntok[k] x D features each. */
    int nprompts, *ntok;
    float **txt;

    /* Detector neck levels (288, 144, 72) and the sine position
     * encoding of the 72 x 72 level. */
    struct neck_level neck[3];
    cl_mem fp0, fp1, fp2, pos;

    /* Geometry encoder: CLS token after final_proj and norm (image
     * independent), its three layers and encode_norm. */
    float geo_in[D];
    struct xlayer geo[3];
    cl_mem genw, genb, geo_out;

    struct xlayer enc[6];
    struct dlayer dec[6];
    float *query, presence[D], refpts[NQ * 4];
    float dnorm_w[D], dnorm_b[D], pnorm_w[D], pnorm_b[D];
    struct hmlp bbox, rpbx, rpby, pres_head, ref_head, mask_embed, pmlp;
    float pmlp_nw[D], pmlp_nb[D];
    float *pproj_w, *pproj_b, *hsproj_w, *hsproj_b;

    /* Segmentation head. */
    struct mha cap;
    cl_mem canw, canb, pdw[2], pdb[2], gnw[2], gnb[2], instw, instb;

    /* Work buffers. */
    cl_mem ex, en, ea, eq, ek, ev, eo, emp, P, pk, pv;
    cl_mem dq, dpos, da, dqq, dk, dv, dout, dffn, rdy, rdx, embq;
    float *h_dq, *h_rdy, *h_rdx, *hs;

    /* Current prompt: its token count (with the geometry token), and
     * whether its pixel embedding is in nka. */
    int cur, np, pixel_ready;
};

/* Loading. */

static void load_mha(struct sam_model *m, const struct pth_file *pf,
                     const char *prefix, struct mha *a)
{
    char name[512];
    const struct pth_tensor *t;
    float *w, *b;
    const size_t msz = (size_t)D * D * sizeof(float), vsz = D * sizeof(float);

    snprintf(name, sizeof(name), "%s.in_proj_weight", prefix);
    t = pth_require(pf, name, 2, (long[]){3 * D, D});
    w = pth_read_f32(pf, t);
    snprintf(name, sizeof(name), "%s.in_proj_bias", prefix);
    t = pth_require(pf, name, 1, (long[]){3 * D});
    b = pth_read_f32(pf, t);
    a->qw = ocl_upload_weight(m->ocl, w, msz / sizeof(float), prefix);
    a->kw = ocl_upload_weight(m->ocl, w + D * D, msz / sizeof(float), prefix);
    a->vw =
        ocl_upload_weight(m->ocl, w + 2 * D * D, msz / sizeof(float), prefix);
    a->qb = ocl_upload(m->ocl, b, vsz, prefix);
    a->kb = ocl_upload(m->ocl, b + D, vsz, prefix);
    a->vb = ocl_upload(m->ocl, b + 2 * D, vsz, prefix);
    G_free(w);
    G_free(b);
    a->ow =
        load_dev_w_fmt(m, pf, 2, (long[]){D, D}, "%s.out_proj.weight", prefix);
    a->ob = load_dev_fmt(m, pf, 1, (long[]){D}, "%s.out_proj.bias", prefix);
}

static void release_mha(struct mha *a)
{
    cl_mem *v[] = {&a->qw, &a->qb, &a->kw, &a->kb,
                   &a->vw, &a->vb, &a->ow, &a->ob};
    size_t i;

    for (i = 0; i < sizeof(v) / sizeof(v[0]); i++)
        ocl_release(v[i]);
}

static cl_mem vec_dev(struct sam_model *m, const struct pth_file *pf,
                      const char *prefix, const char *what)
{
    return load_dev_fmt(m, pf, 1, (long[]){D}, "%s.%s", prefix, what);
}

static void load_xlayer(struct sam_model *m, const struct pth_file *pf,
                        const char *prefix, const char *ca, struct xlayer *l)
{
    char name[1024];

    snprintf(name, sizeof(name), "%s.self_attn", prefix);
    load_mha(m, pf, name, &l->sa);
    snprintf(name, sizeof(name), "%s.%s", prefix, ca);
    load_mha(m, pf, name, &l->ca);
    l->n1w = vec_dev(m, pf, prefix, "norm1.weight");
    l->n1b = vec_dev(m, pf, prefix, "norm1.bias");
    l->n2w = vec_dev(m, pf, prefix, "norm2.weight");
    l->n2b = vec_dev(m, pf, prefix, "norm2.bias");
    l->n3w = vec_dev(m, pf, prefix, "norm3.weight");
    l->n3b = vec_dev(m, pf, prefix, "norm3.bias");
    l->l1w =
        load_dev_w_fmt(m, pf, 2, (long[]){FFN, D}, "%s.linear1.weight", prefix);
    l->l1b = load_dev_fmt(m, pf, 1, (long[]){FFN}, "%s.linear1.bias", prefix);
    l->l2w =
        load_dev_w_fmt(m, pf, 2, (long[]){D, FFN}, "%s.linear2.weight", prefix);
    l->l2b = vec_dev(m, pf, prefix, "linear2.bias");
}

static void release_xlayer(struct xlayer *l)
{
    cl_mem *v[] = {&l->n1w, &l->n1b, &l->n2w, &l->n2b, &l->n3w,
                   &l->n3b, &l->l1w, &l->l1b, &l->l2w, &l->l2b};
    size_t i;

    release_mha(&l->sa);
    release_mha(&l->ca);
    for (i = 0; i < sizeof(v) / sizeof(v[0]); i++)
        ocl_release(v[i]);
}

static void load_dlayer(struct sam_model *m, const struct pth_file *pf,
                        const char *prefix, struct dlayer *l)
{
    char name[1024];

    snprintf(name, sizeof(name), "%s.self_attn", prefix);
    load_mha(m, pf, name, &l->sa);
    snprintf(name, sizeof(name), "%s.ca_text", prefix);
    load_mha(m, pf, name, &l->ct);
    snprintf(name, sizeof(name), "%s.cross_attn", prefix);
    load_mha(m, pf, name, &l->ca);
    l->n1w = vec_dev(m, pf, prefix, "norm1.weight");
    l->n1b = vec_dev(m, pf, prefix, "norm1.bias");
    l->n2w = vec_dev(m, pf, prefix, "norm2.weight");
    l->n2b = vec_dev(m, pf, prefix, "norm2.bias");
    l->n3w = vec_dev(m, pf, prefix, "norm3.weight");
    l->n3b = vec_dev(m, pf, prefix, "norm3.bias");
    l->ctw = vec_dev(m, pf, prefix, "catext_norm.weight");
    l->ctb = vec_dev(m, pf, prefix, "catext_norm.bias");
    l->l1w =
        load_dev_w_fmt(m, pf, 2, (long[]){FFN, D}, "%s.linear1.weight", prefix);
    l->l1b = load_dev_fmt(m, pf, 1, (long[]){FFN}, "%s.linear1.bias", prefix);
    l->l2w =
        load_dev_w_fmt(m, pf, 2, (long[]){D, FFN}, "%s.linear2.weight", prefix);
    l->l2b = vec_dev(m, pf, prefix, "linear2.bias");
}

static void release_dlayer(struct dlayer *l)
{
    cl_mem *v[] = {&l->n1w, &l->n1b, &l->n2w, &l->n2b, &l->n3w, &l->n3b,
                   &l->ctw, &l->ctb, &l->l1w, &l->l1b, &l->l2w, &l->l2b};
    size_t i;

    release_mha(&l->sa);
    release_mha(&l->ct);
    release_mha(&l->ca);
    for (i = 0; i < sizeof(v) / sizeof(v[0]); i++)
        ocl_release(v[i]);
}

static void load_hmlp(const struct pth_file *pf, const char *prefix, int nl,
                      struct hmlp *h)
{
    char name[512];
    int i;

    h->nl = nl;
    for (i = 0; i < nl; i++) {
        const struct pth_tensor *t;

        snprintf(name, sizeof(name), "%s.layers.%d.weight", prefix, i);
        t = pth_require(pf, name, 2, NULL);
        h->out[i] = (int)t->shape[0];
        h->in[i] = (int)t->shape[1];
        h->w[i] = pth_read_f32(pf, t);
        snprintf(name, sizeof(name), "%s.layers.%d.bias", prefix, i);
        h->b[i] = load_host(pf, name, 1, (long[]){h->out[i]});
    }
}

static void free_hmlp(struct hmlp *h)
{
    int i;

    for (i = 0; i < h->nl; i++) {
        G_free(h->w[i]);
        G_free(h->b[i]);
    }
}

/* Host MLP with ReLU between layers; y holds out[nl - 1] floats. */
static void run_hmlp(const struct hmlp *h, const float *x, float *y)
{
    float a[FFN], b[FFN];
    const float *in = x;
    int i;

    for (i = 0; i < h->nl; i++) {
        float *out = i == h->nl - 1 ? y : (i % 2 ? b : a);

        host_linear(h->w[i], h->b[i], in, h->in[i], h->out[i], i < h->nl - 1,
                    out);
        in = out;
    }
}

static void host_ln(const float *x, int n, const float *w, const float *b,
                    float *y)
{
    double mean = 0.0, var = 0.0;
    int i;

    for (i = 0; i < n; i++)
        mean += x[i];
    mean /= n;
    for (i = 0; i < n; i++)
        var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    for (i = 0; i < n; i++)
        y[i] = (float)((x[i] - mean) / sqrt(var + EPS)) * w[i] + b[i];
}

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* model_misc.inverse_sigmoid with eps = 1e-3. */
static float inverse_sigmoid(float x)
{
    float x1, x2;

    x = x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x;
    x1 = x < 1e-3f ? 1e-3f : x;
    x2 = 1.0f - x < 1e-3f ? 1e-3f : 1.0f - x;
    return logf(x1 / x2);
}

/* Device attention, GEMM based: heads of width hd, Nq queries over Nk
 * keys, Q/K/V/O rows with their own leading dimensions, scores in
 * m->scores, an optional causal mask or box relative position bias. */
static void attend(struct sam3_text *t, int Nq, int Nk, int heads, int hd,
                   cl_mem Q, int ldq, cl_mem K, int ldk, cl_mem V, int ldv,
                   cl_mem O, int ldo, int causal, cl_mem dy, cl_mem dx, int W)
{
    struct sam_model *m = t->m;
    const long plane = (long)Nq * Nk;
    long hcl = m->score_cap / plane;
    const int hc = hcl > heads ? heads : (int)hcl;
    int h0;

    if (hc < 1)
        G_fatal_error(_("OpenCL device cannot hold %d x %d attention scores"),
                      Nq, Nk);
    for (h0 = 0; h0 < heads; h0 += hc) {
        int hcc = heads - h0 < hc ? heads - h0 : hc;
        cl_kernel k = t->k_soft;
        cl_int cL = Nk, cNq = Nq, ch0 = h0, cc = causal, cW = W, cnh = heads;
        size_t ls = Nk >= 256 ? 256 : 64, gs = (size_t)hcc * Nq * ls;

        gemm(m, Nq, Nk, hd, bmat(Q, (long)h0 * hd, ldq, 0, hd),
             bmat(K, (long)h0 * hd, ldk, 0, hd), 1,
             bmat(m->scores, 0, Nk, 0, plane), hcc, hcc, NULL, NULL, 0,
             1.0f / sqrtf((float)hd), ACT_NONE);
        SETARG(k, 0, m->scores);
        SETARG(k, 1, cL);
        SETARG(k, 2, cNq);
        SETARG(k, 3, ch0);
        SETARG(k, 4, cc);
        SETARG(k, 5, dy);
        SETARG(k, 6, dx);
        SETARG(k, 7, cW);
        SETARG(k, 8, cnh);
        run(m, k, 1, &gs, &ls, "softmax_ext");
        gemm(m, Nq, hd, Nk, bmat(m->scores, 0, Nk, 0, plane),
             bmat(V, (long)h0 * hd, ldv, 0, hd), 0,
             bmat(O, (long)h0 * hd, ldo, 0, hd), hcc, hcc, NULL, NULL, 0, 1.0f,
             ACT_NONE);
    }
}

/* Detector multi-head attention: out += out_proj(attn(q_in Wq, k_in Wk,
 * v_in Wv)), with the q/k/v/o scratch buffers given. */
static void mha_fwd(struct sam3_text *t, const struct mha *a, int Nq, int Nk,
                    cl_mem qin, cl_mem kin, cl_mem vin, cl_mem out, cl_mem q,
                    cl_mem k, cl_mem v, cl_mem o, cl_mem dy, cl_mem dx)
{
    struct sam_model *m = t->m;

    linear(m, Nq, D, D, qin, 0, a->qw, a->qb, q, 0, ACT_NONE, 0);
    linear(m, Nk, D, D, kin, 0, a->kw, a->kb, k, 0, ACT_NONE, 0);
    linear(m, Nk, D, D, vin, 0, a->vw, a->vb, v, 0, ACT_NONE, 0);
    attend(t, Nq, Nk, NH, HD, q, D, k, D, v, D, o, D, 0, dy, dx, SAM_GRID);
    linear(m, Nq, D, D, o, 0, a->ow, a->ob, out, 0, ACT_NONE, 1);
}

static void add(struct sam_model *m, cl_mem out, cl_mem a, cl_mem b, long n)
{
    add_bcast(m, out, a, b, n, n);
}

static void copy_buf(struct sam_model *m, cl_mem src, size_t soff, cl_mem dst,
                     size_t doff, size_t n)
{
    ocl_check(clEnqueueCopyBuffer(m->ocl->queue, src, dst, soff, doff, n, 0,
                                  NULL, NULL),
              "clEnqueueCopyBuffer");
}

/* Text encoder (VETextEncoder): 24 causal pre-norm blocks of width
 * 1024 on the token and position embeddings, ln_final and the resizer
 * to D channels. Its weights are only kept while the prompts are
 * encoded. */
static void encode_prompts(struct sam3_text *t, const struct pth_file *pf,
                           char **prompts)
{
    struct sam_model *m = t->m;
    struct ocl_backend *ocl = m->ocl;
    struct bpe_tokenizer *tk = tokenizer_load();
    const struct pth_tensor *te, *tp;
    const char *p = TXT ".encoder.transformer.resblocks";
    char name[512];
    long W, Fw;
    int L, l, k, ids[TOK_CONTEXT];
    struct tblock {
        cl_mem n1w, n1b, qw, qb, kw, kb, vw, vb, ow, ob, n2w, n2b, fw, fb, pw,
            pb;
    } *blk;
    cl_mem lfw, lfb, rw, rb, tx, tn, tq, tk_, tv, to, th, tr;
    float *pe, *x;

    te = pth_require(pf, TXT ".encoder.token_embedding.weight", 2, NULL);
    W = te->shape[1];
    tp = pth_require(pf, TXT ".encoder.positional_embedding", 2,
                     (long[]){TOK_CONTEXT, W});
    for (L = 0;; L++) {
        snprintf(name, sizeof(name), "%s.%d.ln_1.weight", p, L);
        if (!pth_find(pf, name))
            break;
    }
    snprintf(name, sizeof(name), "%s.0.mlp.c_fc.weight", p);
    Fw = pth_require(pf, name, 2, (long[]){-1, W})->shape[0];
    if (W % 64 != 0)
        G_fatal_error(_("Unsupported text encoder width %ld"), W);
    G_message(_("Encoding %d text prompt(s) (text encoder: width %ld, %d "
                "blocks)..."),
              t->nprompts, W, L);

    blk = G_calloc(L, sizeof(*blk));
    for (l = 0; l < L; l++) {
        struct tblock *b = &blk[l];
        const size_t msz = (size_t)W * W * sizeof(float),
                     vsz = W * sizeof(float);
        float *w, *bb;

        snprintf(name, sizeof(name), "%s.%d.attn.in_proj_weight", p, l);
        w = load_host(pf, name, 2, (long[]){3 * W, W});
        snprintf(name, sizeof(name), "%s.%d.attn.in_proj_bias", p, l);
        bb = load_host(pf, name, 1, (long[]){3 * W});
        b->qw = ocl_upload_weight(ocl, w, msz / sizeof(float), name);
        b->kw = ocl_upload_weight(ocl, w + W * W, msz / sizeof(float), name);
        b->vw =
            ocl_upload_weight(ocl, w + 2 * W * W, msz / sizeof(float), name);
        b->qb = ocl_upload(ocl, bb, vsz, name);
        b->kb = ocl_upload(ocl, bb + W, vsz, name);
        b->vb = ocl_upload(ocl, bb + 2 * W, vsz, name);
        G_free(w);
        G_free(bb);
        b->ow = load_dev_w_fmt(m, pf, 2, (long[]){W, W},
                             "%s.%d.attn.out_proj.weight", p, l);
        b->ob = load_dev_fmt(m, pf, 1, (long[]){W}, "%s.%d.attn.out_proj.bias",
                             p, l);
        b->n1w = load_dev_fmt(m, pf, 1, (long[]){W}, "%s.%d.ln_1.weight", p, l);
        b->n1b = load_dev_fmt(m, pf, 1, (long[]){W}, "%s.%d.ln_1.bias", p, l);
        b->n2w = load_dev_fmt(m, pf, 1, (long[]){W}, "%s.%d.ln_2.weight", p, l);
        b->n2b = load_dev_fmt(m, pf, 1, (long[]){W}, "%s.%d.ln_2.bias", p, l);
        b->fw = load_dev_w_fmt(m, pf, 2, (long[]){Fw, W}, "%s.%d.mlp.c_fc.weight",
                             p, l);
        b->fb =
            load_dev_fmt(m, pf, 1, (long[]){Fw}, "%s.%d.mlp.c_fc.bias", p, l);
        b->pw = load_dev_w_fmt(m, pf, 2, (long[]){W, Fw},
                             "%s.%d.mlp.c_proj.weight", p, l);
        b->pb =
            load_dev_fmt(m, pf, 1, (long[]){W}, "%s.%d.mlp.c_proj.bias", p, l);
    }
    lfw = load_dev(m, pf, TXT ".encoder.ln_final.weight", 1, (long[]){W});
    lfb = load_dev(m, pf, TXT ".encoder.ln_final.bias", 1, (long[]){W});
    rw = load_dev_w(m, pf, TXT ".resizer.weight", 2, (long[]){D, W});
    rb = load_dev(m, pf, TXT ".resizer.bias", 1, (long[]){D});

    tx = ocl_alloc(ocl, TOK_CONTEXT * W * sizeof(float), "text tokens");
    tn = ocl_alloc(ocl, TOK_CONTEXT * W * sizeof(float), "text normalized");
    tq = ocl_alloc(ocl, TOK_CONTEXT * W * sizeof(float), "text q");
    tk_ = ocl_alloc(ocl, TOK_CONTEXT * W * sizeof(float), "text k");
    tv = ocl_alloc(ocl, TOK_CONTEXT * W * sizeof(float), "text v");
    to = ocl_alloc(ocl, TOK_CONTEXT * W * sizeof(float), "text attention");
    th = ocl_alloc(ocl, TOK_CONTEXT * Fw * sizeof(float), "text mlp");
    tr = ocl_alloc(ocl, TOK_CONTEXT * D * sizeof(float), "text features");
    pe = pth_read_f32(pf, tp);
    x = G_malloc(TOK_CONTEXT * W * sizeof(float));

    t->ntok = G_calloc(t->nprompts, sizeof(int));
    t->txt = G_calloc(t->nprompts, sizeof(float *));
    for (k = 0; k < t->nprompts; k++) {
        int n = tokenizer_encode(tk, prompts[k], ids), i, j;

        G_verbose_message(_("Prompt <%s>: %d tokens"), prompts[k], n);
        /* Padding is never attended by the real tokens (causal mask) and
         * is masked out downstream, so only the n real tokens are run. */
        for (i = 0; i < n; i++) {
            pth_read_row_f32(pf, te, ids[i], x + (size_t)i * W);
            for (j = 0; j < W; j++)
                x[(size_t)i * W + j] += pe[(size_t)i * W + j];
        }
        ocl_write(ocl, tx, 0, x, (size_t)n * W * sizeof(float));
        for (l = 0; l < L; l++) {
            const struct tblock *b = &blk[l];

            layernorm(m, n, (int)W, tx, tn, b->n1w, b->n1b, EPS, ACT_NONE);
            linear(m, n, (int)W, (int)W, tn, 0, b->qw, b->qb, tq, 0, ACT_NONE,
                   0);
            linear(m, n, (int)W, (int)W, tn, 0, b->kw, b->kb, tk_, 0, ACT_NONE,
                   0);
            linear(m, n, (int)W, (int)W, tn, 0, b->vw, b->vb, tv, 0, ACT_NONE,
                   0);
            attend(t, n, n, (int)(W / 64), 64, tq, (int)W, tk_, (int)W, tv,
                   (int)W, to, (int)W, 1, NULL, NULL, 0);
            linear(m, n, (int)W, (int)W, to, 0, b->ow, b->ob, tx, 0, ACT_NONE,
                   1);
            layernorm(m, n, (int)W, tx, tn, b->n2w, b->n2b, EPS, ACT_NONE);
            linear(m, n, (int)W, (int)Fw, tn, 0, b->fw, b->fb, th, 0, ACT_GELU,
                   0);
            linear(m, n, (int)Fw, (int)W, th, 0, b->pw, b->pb, tx, 0, ACT_NONE,
                   1);
        }
        layernorm(m, n, (int)W, tx, tn, lfw, lfb, EPS, ACT_NONE);
        linear(m, n, (int)W, D, tn, 0, rw, rb, tr, 0, ACT_NONE, 0);
        t->ntok[k] = n;
        t->txt[k] = G_malloc((size_t)n * D * sizeof(float));
        ocl_read(ocl, tr, 0, t->txt[k], (size_t)n * D * sizeof(float));
    }

    G_free(pe);
    G_free(x);
    {
        cl_mem *v[] = {&lfw, &lfb, &rw, &rb, &tx, &tn,
                       &tq,  &tk_, &tv, &to, &th, &tr};
        size_t i;

        for (i = 0; i < sizeof(v) / sizeof(v[0]); i++)
            ocl_release(v[i]);
    }
    for (l = 0; l < L; l++) {
        struct tblock *b = &blk[l];
        cl_mem *v[] = {&b->n1w, &b->n1b, &b->qw, &b->qb, &b->kw,  &b->kb,
                       &b->vw,  &b->vb,  &b->ow, &b->ob, &b->n2w, &b->n2b,
                       &b->fw,  &b->fb,  &b->pw, &b->pb};
        size_t i;

        for (i = 0; i < sizeof(v) / sizeof(v[0]); i++)
            ocl_release(v[i]);
    }
    G_free(blk);
    tokenizer_free(tk);
}

/* PositionEmbeddingSine(256, normalize=True) of the 72 x 72 grid,
 * token-major: 128 y channels then 128 x channels. */
static float *sine_pos(void)
{
    float *pos = G_malloc((size_t)T72 * D * sizeof(float));
    const int nf = D / 2;
    int y, x, i;

    for (y = 0; y < SAM_GRID; y++)
        for (x = 0; x < SAM_GRID; x++) {
            float *p = pos + ((size_t)y * SAM_GRID + x) * D;
            const double ye = (y + 1) / (SAM_GRID + 1e-6) * 2.0 * M_PI;
            const double xe = (x + 1) / (SAM_GRID + 1e-6) * 2.0 * M_PI;

            for (i = 0; i < nf; i++) {
                double dim_t = pow(10000.0, 2.0 * (i / 2) / nf);

                p[i] = (float)(i % 2 ? cos(ye / dim_t) : sin(ye / dim_t));
                p[nf + i] = (float)(i % 2 ? cos(xe / dim_t) : sin(xe / dim_t));
            }
        }
    return pos;
}

struct sam3_text *text_load(struct sam_model *m, const struct pth_file *pf,
                            int n, char **prompts)
{
    struct sam3_text *t = G_calloc(1, sizeof(*t));
    struct ocl_backend *ocl = m->ocl;
    const size_t f = sizeof(float);
    char name[512];
    float *pos, cls[D];
    int i;

    t->m = m;
    t->nprompts = n;
    t->k_soft = ocl_kernel(ocl, "softmax_ext");
    t->k_gn = ocl_kernel(ocl, "groupnorm_relu");
    t->k_up2 = ocl_kernel(ocl, "up2_add");
    t->cur = -1;

    encode_prompts(t, pf, prompts);

    G_message(_("Loading the SAM 3 concept detector..."));
    load_neck(m, pf, "detector.backbone.vision_backbone.convs", t->neck);
    pos = sine_pos();
    t->pos = ocl_upload(ocl, pos, (size_t)T72 * D * f, "sine position");
    G_free(pos);

    /* Geometry encoder with an empty prompt: only the CLS token, through
     * final_proj and norm (image independent), then three layers. */
    load_host_into(pf, DET ".geometry_encoder.cls_embed.weight", 2,
                   (long[]){1, D}, cls);
    {
        float *w = load_host(pf, DET ".geometry_encoder.final_proj.weight", 2,
                             (long[]){D, D});
        float *b = load_host(pf, DET ".geometry_encoder.final_proj.bias", 1,
                             (long[]){D});
        float nw[D], nb[D], y[D];

        load_host_into(pf, DET ".geometry_encoder.norm.weight", 1, (long[]){D},
                       nw);
        load_host_into(pf, DET ".geometry_encoder.norm.bias", 1, (long[]){D},
                       nb);
        host_linear(w, b, cls, D, D, 0, y);
        host_ln(y, D, nw, nb, t->geo_in);
        G_free(w);
        G_free(b);
    }
    for (i = 0; i < 3; i++) {
        snprintf(name, sizeof(name), DET ".geometry_encoder.encode.%d", i);
        load_xlayer(m, pf, name, "cross_attn_image", &t->geo[i]);
    }
    t->genw = vec_dev(m, pf, DET ".geometry_encoder", "encode_norm.weight");
    t->genb = vec_dev(m, pf, DET ".geometry_encoder", "encode_norm.bias");

    for (i = 0; i < 6; i++) {
        snprintf(name, sizeof(name), DET ".transformer.encoder.layers.%d", i);
        load_xlayer(m, pf, name, "cross_attn_image", &t->enc[i]);
        snprintf(name, sizeof(name), DET ".transformer.decoder.layers.%d", i);
        load_dlayer(m, pf, name, &t->dec[i]);
    }
    t->query = load_host(pf, DET ".transformer.decoder.query_embed.weight", 2,
                         (long[]){NQ, D});
    load_host_into(pf, DET ".transformer.decoder.reference_points.weight", 2,
                   (long[]){NQ, 4}, t->refpts);
    load_host_into(pf, DET ".transformer.decoder.presence_token.weight", 2,
                   (long[]){1, D}, t->presence);
    load_host_into(pf, DET ".transformer.decoder.norm.weight", 1, (long[]){D},
                   t->dnorm_w);
    load_host_into(pf, DET ".transformer.decoder.norm.bias", 1, (long[]){D},
                   t->dnorm_b);
    load_host_into(pf,
                   DET ".transformer.decoder.presence_token_out_norm.weight", 1,
                   (long[]){D}, t->pnorm_w);
    load_host_into(pf, DET ".transformer.decoder.presence_token_out_norm.bias",
                   1, (long[]){D}, t->pnorm_b);
    load_hmlp(pf, DET ".transformer.decoder.bbox_embed", 3, &t->bbox);
    load_hmlp(pf, DET ".transformer.decoder.boxRPB_embed_x", 2, &t->rpbx);
    load_hmlp(pf, DET ".transformer.decoder.boxRPB_embed_y", 2, &t->rpby);
    load_hmlp(pf, DET ".transformer.decoder.presence_token_head", 3,
              &t->pres_head);
    load_hmlp(pf, DET ".transformer.decoder.ref_point_head", 2, &t->ref_head);
    if (t->rpbx.out[1] != NH || t->rpby.out[1] != NH ||
        t->ref_head.in[0] != 2 * D || t->bbox.out[2] != 4)
        G_fatal_error(_("Unexpected SAM 3 decoder heads in <%s>"), pf->path);

    load_hmlp(pf, DET ".dot_prod_scoring.prompt_mlp", 2, &t->pmlp);
    load_host_into(pf, DET ".dot_prod_scoring.prompt_mlp.out_norm.weight", 1,
                   (long[]){D}, t->pmlp_nw);
    load_host_into(pf, DET ".dot_prod_scoring.prompt_mlp.out_norm.bias", 1,
                   (long[]){D}, t->pmlp_nb);
    t->pproj_w = load_host(pf, DET ".dot_prod_scoring.prompt_proj.weight", 2,
                           (long[]){D, D});
    t->pproj_b =
        load_host(pf, DET ".dot_prod_scoring.prompt_proj.bias", 1, (long[]){D});
    t->hsproj_w = load_host(pf, DET ".dot_prod_scoring.hs_proj.weight", 2,
                            (long[]){D, D});
    t->hsproj_b =
        load_host(pf, DET ".dot_prod_scoring.hs_proj.bias", 1, (long[]){D});

    load_mha(m, pf, DET ".segmentation_head.cross_attend_prompt", &t->cap);
    t->canw =
        vec_dev(m, pf, DET ".segmentation_head", "cross_attn_norm.weight");
    t->canb = vec_dev(m, pf, DET ".segmentation_head", "cross_attn_norm.bias");
    for (i = 0; i < 2; i++) {
        t->pdw[i] = load_dev_fmt(
            m, pf, 4, (long[]){D, D, 3, 3},
            DET ".segmentation_head.pixel_decoder.conv_layers.%d.weight", i);
        t->pdb[i] = load_dev_fmt(
            m, pf, 1, (long[]){D},
            DET ".segmentation_head.pixel_decoder.conv_layers.%d.bias", i);
        t->gnw[i] = load_dev_fmt(
            m, pf, 1, (long[]){D},
            DET ".segmentation_head.pixel_decoder.norms.%d.weight", i);
        t->gnb[i] = load_dev_fmt(
            m, pf, 1, (long[]){D},
            DET ".segmentation_head.pixel_decoder.norms.%d.bias", i);
    }
    t->instw =
        load_dev(m, pf, DET ".segmentation_head.instance_seg_head.weight", 4,
                 (long[]){D, D, 1, 1});
    t->instb = load_dev(m, pf, DET ".segmentation_head.instance_seg_head.bias",
                        1, (long[]){D});
    load_hmlp(pf, DET ".segmentation_head.mask_predictor.mask_embed", 3,
              &t->mask_embed);

    t->fp0 = ocl_alloc(ocl, (size_t)T288 * D * f, "detector level 0");
    t->fp1 = ocl_alloc(ocl, (size_t)4 * T72 * D * f, "detector level 1");
    t->fp2 = ocl_alloc(ocl, (size_t)T72 * D * f, "detector level 2");
    {
        cl_mem *v[] = {&t->ex, &t->en, &t->ea, &t->eq,
                       &t->ek, &t->ev, &t->eo, &t->emp};

        for (i = 0; i < (int)(sizeof(v) / sizeof(v[0])); i++)
            *v[i] = ocl_alloc(ocl, (size_t)T72 * D * f, "detector encoder");
    }
    t->P = ocl_alloc(ocl, (TOK_CONTEXT + 1) * D * f, "prompt tokens");
    t->pk = ocl_alloc(ocl, (TOK_CONTEXT + 1) * D * f, "prompt keys");
    t->pv = ocl_alloc(ocl, (TOK_CONTEXT + 1) * D * f, "prompt values");
    t->geo_out = ocl_alloc(ocl, D * f, "geometry token");
    {
        cl_mem *v[] = {&t->dq, &t->dpos, &t->da,  &t->dqq,
                       &t->dk, &t->dv,   &t->dout};

        for (i = 0; i < (int)(sizeof(v) / sizeof(v[0])); i++)
            *v[i] = ocl_alloc(ocl, (size_t)NQP * D * f, "decoder queries");
    }
    t->dffn = ocl_alloc(ocl, (size_t)NQP * FFN * f, "decoder mlp");
    t->rdy = ocl_alloc(ocl, (size_t)NQP * SAM_GRID * NH * f, "box bias y");
    t->rdx = ocl_alloc(ocl, (size_t)NQP * SAM_GRID * NH * f, "box bias x");
    t->embq = ocl_alloc(ocl, (size_t)3 * m->batch * D * f, "mask embeddings");
    t->h_dq = G_malloc((size_t)NQP * D * f);
    t->h_rdy = G_calloc((size_t)NQP * SAM_GRID * NH, f);
    t->h_rdx = G_calloc((size_t)NQP * SAM_GRID * NH, f);
    t->hs = G_malloc((size_t)NQ * D * f);
    return t;
}

/* Per image. */

void text_set_image(struct sam3_text *t)
{
    struct sam_model *m = t->m;
    int i;

    neck_level(m, &t->neck[0], t->fp0);
    neck_level(m, &t->neck[1], t->fp1);
    neck_level(m, &t->neck[2], t->fp2);

    /* Geometry encoder on the CLS token: self-attention over one token
     * is its value projection; cross-attention to the 72 x 72 detector
     * features with the position encoding on the keys. */
    add(m, t->emp, t->fp2, t->pos, (long)T72 * D);
    ocl_write(m->ocl, t->geo_out, 0, t->geo_in, D * sizeof(float));
    for (i = 0; i < 3; i++) {
        const struct xlayer *l = &t->geo[i];

        layernorm(m, 1, D, t->geo_out, t->dqq, l->n1w, l->n1b, EPS, ACT_NONE);
        linear(m, 1, D, D, t->dqq, 0, l->sa.vw, l->sa.vb, t->dv, 0, ACT_NONE,
               0);
        linear(m, 1, D, D, t->dv, 0, l->sa.ow, l->sa.ob, t->geo_out, 0,
               ACT_NONE, 1);
        layernorm(m, 1, D, t->geo_out, t->dqq, l->n2w, l->n2b, EPS, ACT_NONE);
        mha_fwd(t, &l->ca, 1, T72, t->dqq, t->emp, t->fp2, t->geo_out, t->dk,
                t->ek, t->ev, t->dout, NULL, NULL);
        layernorm(m, 1, D, t->geo_out, t->dqq, l->n3w, l->n3b, EPS, ACT_NONE);
        linear(m, 1, D, FFN, t->dqq, 0, l->l1w, l->l1b, t->dffn, 0, ACT_RELU,
               0);
        linear(m, 1, FFN, D, t->dffn, 0, l->l2w, l->l2b, t->geo_out, 0,
               ACT_NONE, 1);
    }
    layernorm(m, 1, D, t->geo_out, t->geo_out, t->genw, t->genb, EPS, ACT_NONE);
    t->cur = -1;
    t->pixel_ready = 0;
}

/* DETR conditional query position: gen_sineembed_for_position of a
 * (cx, cy, w, h) box, 128 channels per coordinate in y, x, w, h order. */
static void sine_box(const float *b, float *out)
{
    const int nf = D / 2;
    const float c[4] = {b[1], b[0], b[2], b[3]};
    int k, i;

    for (k = 0; k < 4; k++)
        for (i = 0; i < nf; i++) {
            double v = c[k] * 2.0 * M_PI / pow(10000.0, 2.0 * (i / 2) / nf);

            out[k * nf + i] = (float)(i % 2 ? cos(v) : sin(v));
        }
}

/* Box relative position bias inputs: sign(d) log2(8 |d| + 1) / 3. */
static float rpb_log(float d)
{
    d *= 8.0f;
    return (float)((d > 0) - (d < 0)) * log2f(fabsf(d) + 1.0f) / 3.0f;
}

/* Fusion encoder, decoder and scoring for prompt k; leaves the decoder
 * output (hs) on the host and the encoder memory in t->ex. */
int text_detect(struct sam3_text *t, int k, float threshold, int *qidx,
                float *score)
{
    struct sam_model *m = t->m;
    const size_t f = sizeof(float);
    const int np = t->ntok[k] + 1;
    float refs[NQ * 4], presence_logit = 0.0f, pooled[D], pp[D];
    int l, q, n = 0;

    t->cur = k;
    t->np = np;
    t->pixel_ready = 0;

    /* Prompt: the text tokens, then the geometry encoder's token. */
    ocl_write(m->ocl, t->P, 0, t->txt[k], (size_t)t->ntok[k] * D * f);
    copy_buf(m, t->geo_out, 0, t->P, (size_t)t->ntok[k] * D * f, D * f);

    /* Fusion encoder: pre-norm layers on the image tokens, position
     * encoding on the self-attention queries and keys. */
    copy_buf(m, t->fp2, 0, t->ex, 0, (size_t)T72 * D * f);
    for (l = 0; l < 6; l++) {
        const struct xlayer *L = &t->enc[l];

        layernorm(m, T72, D, t->ex, t->en, L->n1w, L->n1b, EPS, ACT_NONE);
        add(m, t->ea, t->en, t->pos, (long)T72 * D);
        mha_fwd(t, &L->sa, T72, T72, t->ea, t->ea, t->en, t->ex, t->eq, t->ek,
                t->ev, t->eo, NULL, NULL);
        layernorm(m, T72, D, t->ex, t->en, L->n2w, L->n2b, EPS, ACT_NONE);
        mha_fwd(t, &L->ca, T72, np, t->en, t->P, t->P, t->ex, t->eq, t->pk,
                t->pv, t->eo, NULL, NULL);
        layernorm(m, T72, D, t->ex, t->en, L->n3w, L->n3b, EPS, ACT_NONE);
        linear(m, T72, D, FFN, t->en, 0, L->l1w, L->l1b, m->mlpbuf, 0, ACT_RELU,
               0);
        linear(m, T72, FFN, D, m->mlpbuf, 0, L->l2w, L->l2b, t->ex, 0, ACT_NONE,
               1);
    }
    /* Decoder keys: the encoder memory with the position encoding. */
    add(m, t->emp, t->ex, t->pos, (long)T72 * D);

    /* Decoder: the presence token, then the 200 queries. */
    memcpy(t->h_dq, t->presence, D * f);
    memcpy(t->h_dq + D, t->query, (size_t)NQ * D * f);
    ocl_write(m->ocl, t->dq, 0, t->h_dq, (size_t)NQP * D * f);
    for (q = 0; q < NQ * 4; q++)
        refs[q] = sigmoidf(t->refpts[q]);

    for (l = 0; l < 6; l++) {
        const struct dlayer *L = &t->dec[l];
        float *qpos = t->h_dq; /* Reused as the query position buffer. */

        /* Query positions from the current boxes (zero for presence)
         * and the box relative position bias of the image attention. */
        memset(qpos, 0, D * f);
#pragma omp parallel for schedule(dynamic, 8)
        for (q = 0; q < NQ; q++) {
            const float *b = refs + 4 * q;
            const float x0 = b[0] - 0.5f * b[2], y0 = b[1] - 0.5f * b[3];
            const float x1 = b[0] + 0.5f * b[2], y1 = b[1] + 0.5f * b[3];
            float sine[2 * D], in[2];
            int c;

            sine_box(b, sine);
            run_hmlp(&t->ref_head, sine, qpos + (size_t)(q + 1) * D);
            for (c = 0; c < SAM_GRID; c++) {
                const float co = (float)c / SAM_GRID;
                const size_t o = ((size_t)(q + 1) * SAM_GRID + c) * NH;

                in[0] = rpb_log(co - x0);
                in[1] = rpb_log(co - x1);
                run_hmlp(&t->rpbx, in, t->h_rdx + o);
                in[0] = rpb_log(co - y0);
                in[1] = rpb_log(co - y1);
                run_hmlp(&t->rpby, in, t->h_rdy + o);
            }
        }
        ocl_write(m->ocl, t->dpos, 0, qpos, (size_t)NQP * D * f);
        ocl_write(m->ocl, t->rdx, 0, t->h_rdx, (size_t)NQP * SAM_GRID * NH * f);
        ocl_write(m->ocl, t->rdy, 0, t->h_rdy, (size_t)NQP * SAM_GRID * NH * f);

        add(m, t->da, t->dq, t->dpos, (long)NQP * D);
        mha_fwd(t, &L->sa, NQP, NQP, t->da, t->da, t->dq, t->dq, t->dqq, t->dk,
                t->dv, t->dout, NULL, NULL);
        layernorm(m, NQP, D, t->dq, t->dq, L->n2w, L->n2b, EPS, ACT_NONE);
        add(m, t->da, t->dq, t->dpos, (long)NQP * D);
        mha_fwd(t, &L->ct, NQP, np, t->da, t->P, t->P, t->dq, t->dqq, t->pk,
                t->pv, t->dout, NULL, NULL);
        layernorm(m, NQP, D, t->dq, t->dq, L->ctw, L->ctb, EPS, ACT_NONE);
        add(m, t->da, t->dq, t->dpos, (long)NQP * D);
        mha_fwd(t, &L->ca, NQP, T72, t->da, t->emp, t->ex, t->dq, t->dqq, t->ek,
                t->ev, t->dout, t->rdy, t->rdx);
        layernorm(m, NQP, D, t->dq, t->dq, L->n1w, L->n1b, EPS, ACT_NONE);
        linear(m, NQP, D, FFN, t->dq, 0, L->l1w, L->l1b, t->dffn, 0, ACT_RELU,
               0);
        linear(m, NQP, FFN, D, t->dffn, 0, L->l2w, L->l2b, t->dq, 0, ACT_NONE,
               1);
        layernorm(m, NQP, D, t->dq, t->dq, L->n3w, L->n3b, EPS, ACT_NONE);

        /* Box refinement on the normalized queries. */
        ocl_read(m->ocl, t->dq, 0, t->h_dq, (size_t)NQP * D * f);
#pragma omp parallel for
        for (q = 0; q < NQ; q++) {
            float *hq = t->hs + (size_t)q * D, delta[4];
            int c;

            host_ln(t->h_dq + (size_t)(q + 1) * D, D, t->dnorm_w, t->dnorm_b,
                    hq);
            run_hmlp(&t->bbox, hq, delta);
            for (c = 0; c < 4; c++)
                refs[4 * q + c] =
                    sigmoidf(inverse_sigmoid(refs[4 * q + c]) + delta[c]);
        }
        {
            float pn[D];

            host_ln(t->h_dq, D, t->pnorm_w, t->pnorm_b, pn);
            run_hmlp(&t->pres_head, pn, &presence_logit);
        }
    }

    /* Dot-product scoring against the mean-pooled prompt. */
    memset(pooled, 0, sizeof(pooled));
    {
        float *pr = G_malloc((size_t)np * D * f);
        int i, c;

        ocl_read(m->ocl, t->P, 0, pr, (size_t)np * D * f);
        for (i = 0; i < np; i++) {
            float y[D], z[D];

            run_hmlp(&t->pmlp, pr + (size_t)i * D, y);
            for (c = 0; c < D; c++)
                y[c] += pr[(size_t)i * D + c];
            host_ln(y, D, t->pmlp_nw, t->pmlp_nb, z);
            for (c = 0; c < D; c++)
                pooled[c] += z[c] / np;
        }
        G_free(pr);
    }
    host_linear(t->pproj_w, t->pproj_b, pooled, D, D, 0, pp);
    {
        const float pres = sigmoidf(presence_logit);

        for (q = 0; q < NQ; q++) {
            float hp[D], s = 0.0f;
            int c;

            host_linear(t->hsproj_w, t->hsproj_b, t->hs + (size_t)q * D, D, D,
                        0, hp);
            for (c = 0; c < D; c++)
                s += hp[c] * pp[c];
            s /= sqrtf((float)D);
            s = s < -12.0f ? -12.0f : s > 12.0f ? 12.0f : s;
            s = sigmoidf(s) * pres;
            if (s > threshold) {
                int j = n++;

                /* Insertion keeps the detections best first. */
                while (j > 0 && score[j - 1] < s) {
                    score[j] = score[j - 1];
                    qidx[j] = qidx[j - 1];
                    j--;
                }
                score[j] = s;
                qidx[j] = q;
            }
        }
    }
    G_debug(1, "prompt %d: presence %.4f, %d detections", k,
            sigmoidf(presence_logit), n);
    return n;
}

int text_mask_capacity(const struct sam3_text *t) { return 3 * t->m->batch; }

/* Segmentation head: the encoder memory attends to the prompt, the
 * pixel decoder merges it with the 144 and 288 detector levels, and the
 * instance head gives per-pixel embeddings (kept in nka). */
static void pixel_embedding(struct sam3_text *t)
{
    struct sam_model *m = t->m;
    const int G = SAM_GRID;
    cl_kernel k;
    cl_int cW, cC = D, total;
    cl_float eps = EPS;
    int s;

    layernorm(m, T72, D, t->ex, t->en, t->canw, t->canb, EPS, ACT_NONE);
    copy_buf(m, t->ex, 0, t->ea, 0, (size_t)T72 * D * sizeof(float));
    mha_fwd(t, &t->cap, T72, t->np, t->en, t->P, t->P, t->ea, t->eq, t->pk,
            t->pv, t->eo, NULL, NULL);

    for (s = 0; s < 2; s++) {
        const int side = (2 << s) * G; /* 144, then 288. */
        cl_mem lo = s ? m->nkb : t->ea, hi = s ? t->fp0 : t->fp1;
        size_t gs = 8 * 256, ls = 256;
        cl_int cHW = side * side;

        k = t->k_up2;
        cW = side;
        total = side * side * D;
        SETARG(k, 0, m->nka);
        SETARG(k, 1, hi);
        SETARG(k, 2, lo);
        SETARG(k, 3, cW);
        SETARG(k, 4, cC);
        SETARG(k, 5, total);
        run1d(m, k, total, "up2_add");
        conv3x3(m, m->nka, side, t->pdw[s], t->pdb[s], m->nkb);
        k = t->k_gn;
        SETARG(k, 0, m->nkb);
        SETARG(k, 1, cHW);
        SETARG(k, 2, cC);
        SETARG(k, 3, t->gnw[s]);
        SETARG(k, 4, t->gnb[s]);
        SETARG(k, 5, eps);
        run(m, k, 1, &gs, &ls, "groupnorm_relu");
    }
    linear(m, T288, D, D, m->nkb, 0, t->instw, t->instb, m->nka, 0, ACT_NONE,
           0);
    t->pixel_ready = 1;
}

void text_masks(struct sam3_text *t, int n, const int *qidx)
{
    struct sam_model *m = t->m;
    float *e = G_malloc((size_t)n * D * sizeof(float));
    int i;

    if (n > text_mask_capacity(t))
        G_fatal_error("text_masks: capacity exceeded");
    if (!t->pixel_ready)
        pixel_embedding(t);
#pragma omp parallel for
    for (i = 0; i < n; i++)
        run_hmlp(&t->mask_embed, t->hs + (size_t)qidx[i] * D,
                 e + (size_t)i * D);
    ocl_write(m->ocl, t->embq, 0, e, (size_t)n * D * sizeof(float));
    G_free(e);
    gemm(m, n, T288, D, mat(t->embq, 0, D), mat(m->nka, 0, D), 1,
         mat(m->lowres, 0, T288), 1, 1, NULL, NULL, 0, 1.0f, ACT_NONE);
}

void text_read_features(struct sam3_text *t, float *img, float *text, int *ntok)
{
    ocl_read(t->m->ocl, t->fp2, 0, img, (size_t)T72 * D * sizeof(float));
    *ntok = t->ntok[0];
    memcpy(text, t->txt[0], (size_t)t->ntok[0] * D * sizeof(float));
}

void text_free(struct sam3_text *t)
{
    int i;

    if (!t)
        return;
    release_neck(t->neck);
    for (i = 0; i < 3; i++)
        release_xlayer(&t->geo[i]);
    for (i = 0; i < 6; i++) {
        release_xlayer(&t->enc[i]);
        release_dlayer(&t->dec[i]);
    }
    release_mha(&t->cap);
    for (i = 0; i < 2; i++) {
        ocl_release(&t->pdw[i]);
        ocl_release(&t->pdb[i]);
        ocl_release(&t->gnw[i]);
        ocl_release(&t->gnb[i]);
    }
    {
        cl_mem *v[] = {&t->fp0,   &t->fp1,     &t->fp2,  &t->pos,  &t->genw,
                       &t->genb,  &t->geo_out, &t->canw, &t->canb, &t->instw,
                       &t->instb, &t->ex,      &t->en,   &t->ea,   &t->eq,
                       &t->ek,    &t->ev,      &t->eo,   &t->emp,  &t->P,
                       &t->pk,    &t->pv,      &t->dq,   &t->dpos, &t->da,
                       &t->dqq,   &t->dk,      &t->dv,   &t->dout, &t->dffn,
                       &t->rdy,   &t->rdx,     &t->embq};
        size_t j;

        for (j = 0; j < sizeof(v) / sizeof(v[0]); j++)
            ocl_release(v[j]);
    }
    free_hmlp(&t->bbox);
    free_hmlp(&t->rpbx);
    free_hmlp(&t->rpby);
    free_hmlp(&t->pres_head);
    free_hmlp(&t->ref_head);
    free_hmlp(&t->mask_embed);
    free_hmlp(&t->pmlp);
    G_free(t->query);
    G_free(t->pproj_w);
    G_free(t->pproj_b);
    G_free(t->hsproj_w);
    G_free(t->hsproj_b);
    for (i = 0; i < t->nprompts; i++)
        G_free(t->txt[i]);
    G_free(t->txt);
    G_free(t->ntok);
    G_free(t->h_dq);
    G_free(t->h_rdy);
    G_free(t->h_rdx);
    G_free(t->hs);
    clReleaseKernel(t->k_soft);
    clReleaseKernel(t->k_gn);
    clReleaseKernel(t->k_up2);
    G_free(t);
}
