/* SPDX-License-Identifier: Unlicense */

#ifndef I_SAM3_OPENCL_SAM3_MODEL_H
#define I_SAM3_OPENCL_SAM3_MODEL_H

#include "ocl_backend.h"
#include "pth_loader.h"

#define SAM_IMG 1008   /* Encoder input side. */
#define SAM_PATCHSZ 14 /* ViT patch side. */
#define SAM_GRID 72    /* Image embedding side (SAM_IMG / 14). */
#define SAM_DIM 256    /* Neck, prompt and decoder embedding width. */
#define SAM_LOWRES 288 /* Low-resolution mask side (4 x SAM_GRID). */
#define SAM_HOLE 256   /* Holes filled in low-resolution masks, pixels. */

struct sam_attn_w {
    cl_mem qw, qb, kw, kb, vw, vb, ow, ob;
    int dint;
};

/* One ViT block of the Perception Encoder backbone. */
struct vit_block {
    cl_mem n1w, n1b, qkvw, qkvb, projw, projb, n2w, n2b;
    cl_mem l1w, l1b, l2w, l2b;
    int window; /* Window side, 0 for global attention. */
};

struct sam_dec_layer {
    struct sam_attn_w self, t2i, i2t;
    cl_mem n1w, n1b, n2w, n2b, n3w, n3b, n4w, n4b;
    cl_mem m1w, m1b, m2w, m2b;
};

/* One level of the neck feeding the point-prompt head (sam2_convs):
 * optional transposed convolutions, then 1x1 and 3x3 convolutions. */
struct neck_level {
    int scale; /* 4, 2 or 1. */
    cl_mem d0w, d0b, d1w, d1b, c1w, c1b, c3w, c3b;
};

struct sam_model {
    struct ocl_backend *ocl;
    int embed, depth, heads, hd, mlp, pt_side;

    /* ViT backbone: patch embedding (no bias), tiled absolute position
     * embedding, ln_pre, blocks, and the RoPE rotations (cos, sin per
     * position and frequency) of windowed and global attention. */
    cl_mem pe_w, pos, lnpw, lnpb;
    struct vit_block *blocks;
    cl_mem rope_win, rope_glob;

    /* Neck, the decoder's high-resolution projections and the no-memory
     * embedding added to the image embedding. */
    struct neck_level neck[3];
    cl_mem s0w, s0b, s1w, s1b;
    cl_mem no_mem;

    /* Prompt encoder (host side, tiny). */
    float gauss[2 * 128];
    float point_emb[4][SAM_DIM];
    float not_a_point[SAM_DIM];
    float no_mask[SAM_DIM];

    /* Mask decoder. */
    struct sam_dec_layer layers[2];
    struct sam_attn_w final_t2i;
    cl_mem nfw, nfb;
    cl_mem up0w, up0b, up1w, up1b, up3w, up3b;
    float obj_token[SAM_DIM], iou_token[SAM_DIM], mask_tokens[4 * SAM_DIM];
    float *hyper_w[4][3], *hyper_b[4][3];
    float *iou_w[3], *iou_b[3];

    cl_kernel k_gemm, k_gemm_big, k_attn_few, k_attn_many, k_ln, k_softmax,
        k_patch, k_add, k_wpart, k_wunpart, k_rope, k_im2col, k_convt, k_logits,
        k_post;

    /* Encoder work buffers. */
    long score_cap; /* Floats of the attention score buffer. */
    cl_mem img, x, xn, xw, qkv, scores, attn, tmp, mlpbuf, nka, nkb, col;
    long col_rows; /* Tokens per band of the 3x3 convolution im2col. */
    cl_mem emb, feat_s0, feat_s1;

    /* Per-image decoder caches: src = embedding + no-mask embedding,
     * dense positional encoding, and the layer-0 projections that do
     * not depend on the prompt. */
    cl_mem src, pe_img, srcpe, kp0, vp0, qp0;

    /* Decoder batch buffers, sized for batch prompts. */
    int batch;
    cl_mem qpe, qry, qin, sq, sk, sv, so, sh;
    cl_mem keys, kin, bq, bk, bv, bo;
    cl_mem u1, g2, u2, hyper, lowres;
    float *h_queries, *h_lowres;

    /* Postprocessed binary masks and their statistics. */
    cl_mem mask, sel, stats;
    size_t mask_cap;
};

/* True if pf holds a SAM 3 checkpoint with the image backbone and the
 * point-prompt (tracker) heads. */
int sam_is_sam3(const struct pth_file *pf);

/* Load the SAM 3 Perception Encoder backbone, the neck of its
 * point-prompt head and that head's prompt encoder and mask decoder
 * onto the device, and allocate work buffers for decoding batch
 * prompts at a time. */
void sam_load(struct sam_model *m, struct ocl_backend *ocl,
              const struct pth_file *pf, int batch);

/* Encode a normalized 3 x 1008 x 1008 CHW image and prepare the
 * prompt-independent decoder caches. */
void sam_encode(struct sam_model *m, const float *img);

/* Copy the encoder outputs to host, token-major: the 72 x 72 x 256
 * image embedding (no-memory embedding included), and the 288 x 288 x
 * 32 and 144 x 144 x 64 high-resolution features. */
void sam_read_features(struct sam_model *m, float *emb, float *s0, float *s1);

/* Decode P <= batch single-point prompts, points given in encoder
 * input pixel coordinates (x, y pairs). Writes the three multimask IoU
 * predictions per prompt to iou[P * 3] and leaves the low-resolution
 * logits on the device. */
void sam_decode(struct sam_model *m, int P, const float *points, float *iou);

/* Postprocess n selected low-resolution masks (index p * 3 + k) to an
 * out_h x out_w crop: holes of at most SAM_HOLE pixels are filled in the
 * low-resolution logits, as SAM3InteractiveImagePredictor does, before
 * the bilinear resize. stats receives 6 ints per mask: count(logit >
 * offset), count(logit > -offset), xmin, ymin, xmax, ymax (xmax < 0 if
 * empty). */
void sam_postprocess(struct sam_model *m, int n, const int *sel, int out_h,
                     int out_w, float offset, int *stats);

/* Read rows y0..y1 and columns x0..x1 (inclusive) of postprocessed mask
 * k into dst (row-major, (x1 - x0 + 1) wide). */
void sam_read_mask(struct sam_model *m, int k, int out_h, int out_w, int x0,
                   int y0, int x1, int y1, unsigned char *dst);

void sam_free(struct sam_model *m);

#endif /* I_SAM3_OPENCL_SAM3_MODEL_H */
