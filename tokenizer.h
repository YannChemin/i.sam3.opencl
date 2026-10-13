/* SPDX-License-Identifier: Unlicense */

#ifndef I_SAM3_OPENCL_TOKENIZER_H
#define I_SAM3_OPENCL_TOKENIZER_H

#define TOK_CONTEXT 32 /* SAM 3 text context length. */

struct bpe_tokenizer;

/* Load CLIP's byte-pair encoding vocabulary (bpe_simple_vocab_16e6.txt.gz),
 * looked up in $I_SAM3_OPENCL_BPE, $GRASS_ADDON_BASE/etc/i.sam3.opencl
 * and $GISBASE/etc/i.sam3.opencl. Fatal if it is not found. */
struct bpe_tokenizer *tokenizer_load(void);

/* Tokenize text as SimpleTokenizer does for SAM 3 (lower-case cleaning,
 * start and end tokens, truncation to TOK_CONTEXT). Writes the token
 * ids to ids[TOK_CONTEXT] and returns their number, padding excluded.
 * Fatal for non-ASCII text, which needs Unicode cleaning the module does
 * not implement. */
int tokenizer_encode(const struct bpe_tokenizer *tk, const char *text,
                     int *ids);

void tokenizer_free(struct bpe_tokenizer *tk);

#endif /* I_SAM3_OPENCL_TOKENIZER_H */
