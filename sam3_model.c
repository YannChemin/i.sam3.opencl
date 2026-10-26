/****************************************************************************
 *
 * MODULE:       i.sam3.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Segment Anything 3 (SAM 3) point-prompt segmentation on
 *               OpenCL: the Perception Encoder ViT backbone, the neck of
 *               the point-prompt head, and that head's prompt encoder and
 *               mask decoder, weights read directly from the original
 *               PyTorch checkpoint.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "progress.h"
#include "pth_loader.h"
#include "sam3_model.h"
#include "sam3_ops.h"

#define ENC_EPS 1e-5f  /* ViT LayerNorm(eps=1e-5). */
#define DEC_EPS 1e-5f  /* nn.LayerNorm default, decoder transformer. */
#define LN2D_EPS 1e-6f /* LayerNorm2d of the decoder's output upscaling. */

/* Object score token, IoU token, 4 mask tokens, point, padding point. */
#define NTOK 8
#define DEC_HEADS 8

#define TRUNK "detector.backbone.vision_backbone.trunk"
#define NECK "detector.backbone.vision_backbone.sam2_convs"

int sam_is_sam3(const struct pth_file *pf)
{
    return pth_find(pf, TRUNK ".pos_embed") &&
           pth_find(pf, "tracker.sam_mask_decoder.iou_token.weight") &&
           pth_find(pf, NECK ".0.conv_1x1.weight");
}

/* Weight loading. */

static size_t loaded_bytes;

cl_mem load_dev(struct sam_model *m, const struct pth_file *pf,
                const char *name, int ndim, const long *shape)
{
    const struct pth_tensor *t = pth_require(pf, name, ndim, shape);
    float *h = pth_read_f32(pf, t);
    size_t n = pth_numel(t) * sizeof(float);
    cl_mem b = ocl_upload(m->ocl, h, n, name);

    loaded_bytes += n;
    G_free(h);
    return b;
}

cl_mem load_dev_w(struct sam_model *m, const struct pth_file *pf,
                  const char *name, int ndim, const long *shape)
{
    const struct pth_tensor *t = pth_require(pf, name, ndim, shape);
    float *h = pth_read_f32(pf, t);
    size_t n = pth_numel(t);
    cl_mem b = ocl_upload_weight(m->ocl, h, n, name);

    loaded_bytes += n * (m->ocl->fp16w ? 2 : sizeof(float));
    G_free(h);
    return b;
}

cl_mem load_dev_w_fmt(struct sam_model *m, const struct pth_file *pf, int ndim,
                      const long *shape, const char *fmt, ...)
{
    char name[256];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    return load_dev_w(m, pf, name, ndim, shape);
}

float *load_host(const struct pth_file *pf, const char *name, int ndim,
                 const long *shape)
{
    return pth_read_f32(pf, pth_require(pf, name, ndim, shape));
}

void load_host_into(const struct pth_file *pf, const char *name, int ndim,
                    const long *shape, float *dst)
{
    const struct pth_tensor *t = pth_require(pf, name, ndim, shape);
    float *h = pth_read_f32(pf, t);

    memcpy(dst, h, pth_numel(t) * sizeof(float));
    G_free(h);
}

cl_mem load_dev_fmt(struct sam_model *m, const struct pth_file *pf, int ndim,
                    const long *shape, const char *fmt, ...)
{
    char name[256];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    return load_dev(m, pf, name, ndim, shape);
}

static void load_attn(struct sam_model *m, const struct pth_file *pf,
                      const char *prefix, int dint, struct sam_attn_w *a)
{
    long w_in[2] = {dint, SAM_DIM}, w_out[2] = {SAM_DIM, dint};
    long b_in[1] = {dint}, b_out[1] = {SAM_DIM};

    a->dint = dint;
    a->qw = load_dev_w_fmt(m, pf, 2, w_in, "%s.q_proj.weight", prefix);
    a->qb = load_dev_fmt(m, pf, 1, b_in, "%s.q_proj.bias", prefix);
    a->kw = load_dev_w_fmt(m, pf, 2, w_in, "%s.k_proj.weight", prefix);
    a->kb = load_dev_fmt(m, pf, 1, b_in, "%s.k_proj.bias", prefix);
    a->vw = load_dev_w_fmt(m, pf, 2, w_in, "%s.v_proj.weight", prefix);
    a->vb = load_dev_fmt(m, pf, 1, b_in, "%s.v_proj.bias", prefix);
    a->ow = load_dev_w_fmt(m, pf, 2, w_out, "%s.out_proj.weight", prefix);
    a->ob = load_dev_fmt(m, pf, 1, b_out, "%s.out_proj.bias", prefix);
}

static void release_attn(struct sam_attn_w *a)
{
    ocl_release(&a->qw);
    ocl_release(&a->qb);
    ocl_release(&a->kw);
    ocl_release(&a->kb);
    ocl_release(&a->vw);
    ocl_release(&a->vb);
    ocl_release(&a->ow);
    ocl_release(&a->ob);
}

/* Kernel launch helpers. */

/* Accumulated kernel times when profiling is enabled. */
static struct {
    const char *what;
    double ms, wall_ms, flops;
    long calls;
} prof[64];
static int nprof;
/* FLOPs of the next launch, set by gemm() for the profile. */
static double pending_flops;

static double now_ms(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

void run(struct sam_model *m, cl_kernel k, int dim, const size_t *g,
         const size_t *l, const char *what)
{
    cl_event ev;
    cl_ulong t0, t1;
    double w0 = 0.0, w1;
    int i;

    /* Wall time as well as event time: some drivers (rusticl on iris,
     * Mesa 25.0) return wrong event timestamps. */
    if (m->ocl->profile) {
        clFinish(m->ocl->queue);
        w0 = now_ms();
    }
    ocl_check(clEnqueueNDRangeKernel(m->ocl->queue, k, dim, NULL, g, l, 0, NULL,
                                     m->ocl->profile ? &ev : NULL),
              what);
    if (!m->ocl->profile) {
        pending_flops = 0.0;
        return;
    }
    clWaitForEvents(1, &ev);
    w1 = now_ms();
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0,
                            NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1,
                            NULL);
    clReleaseEvent(ev);
    for (i = 0; i < nprof && strcmp(prof[i].what, what) != 0; i++)
        ;

    if (i == nprof && nprof < 64)
        prof[nprof++].what = what;
    if (i < 64) {
        prof[i].ms += (t1 - t0) * 1e-6;
        prof[i].wall_ms += w1 - w0;
        prof[i].flops += pending_flops;
        prof[i].calls++;
    }
    pending_flops = 0.0;
}

void run1d(struct sam_model *m, cl_kernel k, size_t n, const char *what)
{
    size_t l = 256, g = (n + l - 1) / l * l;

    run(m, k, 1, &g, &l, what);
}

struct opnd mat(cl_mem buf, long off, int ld)
{
    struct opnd o = {buf, off, ld, 0, 0};

    return o;
}

struct opnd bmat(cl_mem buf, long off, int ld, long s0, long s1)
{
    struct opnd o = {buf, off, ld, s0, s1};

    return o;
}

void gemm(struct sam_model *m, int M, int N, int K, struct opnd A,
          struct opnd B, int transb, struct opnd C, int nb, int nb1,
          cl_mem bias, cl_mem R, long roff, float alpha, int act)
{
    cl_int cM = M, cN = N, cK = K, ct = transb, cnb1 = nb1, cact = act;
    cl_long croff = roff;
    cl_float calpha = alpha;
    size_t g[3], l[3] = {16, 16, 1};
    /* The 128 x 128 float4 kernel needs every A/B row start 16-byte
     * aligned; small or misaligned products use the generic kernel. */
    int big = M >= 128 && N >= 64 && K % 4 == 0 && A.ld % 4 == 0 &&
              A.off % 4 == 0 && A.s0 % 4 == 0 && A.s1 % 4 == 0 &&
              B.ld % 4 == 0 && B.off % 4 == 0 && B.s0 % 4 == 0 && B.s1 % 4 == 0;
    int tile = big ? 128 : 64;
    int half = m->ocl->fp16w && ocl_is_half(B.buf);
    int cpu = m->ocl->arch == OCL_ARCH_CPU;
    cl_kernel k;
    const char *what;

    if (cpu) {
        k = transb ? (half ? m->k_gemm_cpu_t_h : m->k_gemm_cpu_t)
                   : (half ? m->k_gemm_cpu_n_h : m->k_gemm_cpu_n);
        what = transb ? "gemm_cpu_t" : "gemm_cpu_n";
    }
    else if (big) {
        k = half ? m->k_gemm_big_h : m->k_gemm_big;
        what = "gemm_big";
    }
    else {
        k = half ? m->k_gemm_h : m->k_gemm;
        what = "gemm";
    }

    SETARG(k, 0, cM);
    SETARG(k, 1, cN);
    SETARG(k, 2, cK);
    SETARG(k, 3, A.buf);
    SETARG(k, 4, A.off);
    SETARG(k, 5, A.ld);
    SETARG(k, 6, A.s0);
    SETARG(k, 7, A.s1);
    SETARG(k, 8, B.buf);
    SETARG(k, 9, B.off);
    SETARG(k, 10, B.ld);
    SETARG(k, 11, B.s0);
    SETARG(k, 12, B.s1);
    SETARG(k, 13, ct);
    SETARG(k, 14, C.buf);
    SETARG(k, 15, C.off);
    SETARG(k, 16, C.ld);
    SETARG(k, 17, C.s0);
    SETARG(k, 18, C.s1);
    SETARG(k, 19, cnb1);
    SETARG(k, 20, bias);
    SETARG(k, 21, R);
    SETARG(k, 22, croff);
    SETARG(k, 23, calpha);
    SETARG(k, 24, cact);
    pending_flops = 2.0 * M * N * K * nb;
    if (cpu) {
        /* One work-item per 2 x 4 (transb) or 4 x 8 output block; the
         * work-group size is left to the driver. */
        g[0] = transb ? (size_t)(N + 3) / 4 : (size_t)(N + 7) / 8;
        g[1] = transb ? (size_t)(M + 1) / 2 : (size_t)(M + 3) / 4;
        g[2] = nb;
        run(m, k, 3, g, NULL, what);
        return;
    }
    g[0] = (size_t)((N + tile - 1) / tile) * 16;
    g[1] = (size_t)((M + tile - 1) / tile) * 16;
    g[2] = nb;
    run(m, k, 3, g, l, what);
}

/* Y[rows, out] = act(X[rows, in] W^T + b) (+ Y when residual). */
void linear(struct sam_model *m, int rows, int in, int out, cl_mem X, long xoff,
            cl_mem W, cl_mem b, cl_mem Y, long yoff, int act, int residual)
{
    gemm(m, rows, out, in, mat(X, xoff, in), mat(W, 0, in), 1,
         mat(Y, yoff, out), 1, 1, b, residual ? Y : NULL, yoff, 1.0f, act);
}

/* Row LayerNorm: 256 items per row for wide rows, one 64-item wavefront
 * per row for the narrow ones. */
void layernorm(struct sam_model *m, long rows, int C, cl_mem X, cl_mem Y,
               cl_mem g, cl_mem b, float eps, int act)
{
    cl_kernel k = m->k_ln;
    cl_int cC = C, cact = act;
    cl_float ceps = eps;
    size_t ls = C >= 1024 ? 256 : 64, gs = rows * ls;

    SETARG(k, 0, X);
    SETARG(k, 1, Y);
    SETARG(k, 2, g);
    SETARG(k, 3, b);
    SETARG(k, 4, cC);
    SETARG(k, 5, ceps);
    SETARG(k, 6, cact);
    run(m, k, 1, &gs, &ls, "layernorm");
}

static void softmax_rows(struct sam_model *m, cl_mem S, long rows, int L)
{
    cl_kernel k = m->k_softmax;
    cl_int cL = L;
    size_t ls = L >= 256 ? 256 : 64, gs = rows * ls;

    SETARG(k, 0, S);
    SETARG(k, 1, cL);
    run(m, k, 1, &gs, &ls, "softmax_rows");
}

void add_bcast(struct sam_model *m, cl_mem out, cl_mem a, cl_mem b, long n,
               long bmod)
{
    cl_kernel k = m->k_add;
    cl_int cn = (cl_int)n, cb = (cl_int)bmod;

    SETARG(k, 0, out);
    SETARG(k, 1, a);
    SETARG(k, 2, b);
    SETARG(k, 3, cn);
    SETARG(k, 4, cb);
    run1d(m, k, n, "add_bcast");
}

/* 2x2 max pooling of nwin s x s grids, C channels from column xoff of
 * rows ldx floats wide, into out (C floats per row). */

static void window_partition(struct sam_model *m, cl_mem X, cl_mem out, int H,
                             int C, int ws, int nwx)
{
    cl_kernel k = m->k_wpart;
    cl_int cH = H, cC = C, cws = ws, cnwx = nwx;
    cl_int total = nwx * nwx * ws * ws * C;

    SETARG(k, 0, X);
    SETARG(k, 1, out);
    SETARG(k, 2, cH);
    SETARG(k, 3, cH);
    SETARG(k, 4, cC);
    SETARG(k, 5, cws);
    SETARG(k, 6, cnwx);
    SETARG(k, 7, total);
    run1d(m, k, total, "window_partition");
}

static void window_unpartition_add(struct sam_model *m, cl_mem X, cl_mem Wd,
                                   int H, int C, int ws, int nwx)
{
    cl_kernel k = m->k_wunpart;
    cl_int cH = H, cC = C, cws = ws, cnwx = nwx;

    SETARG(k, 0, X);
    SETARG(k, 1, Wd);
    SETARG(k, 2, cH);
    SETARG(k, 3, cH);
    SETARG(k, 4, cC);
    SETARG(k, 5, cws);
    SETARG(k, 6, cnwx);
    run1d(m, k, (size_t)H * H * C, "window_unpartition_add");
}

void convt_scatter(struct sam_model *m, cl_mem G, cl_mem out, cl_mem bias,
                   cl_mem add, int P, int H, int W, int Co, int act)
{
    cl_kernel k = m->k_convt;
    cl_int cH = H, cW = W, cCo = Co, cact = act;
    cl_int total = P * 4 * H * W * Co;

    SETARG(k, 0, G);
    SETARG(k, 1, out);
    SETARG(k, 2, bias);
    SETARG(k, 3, add);
    SETARG(k, 4, cH);
    SETARG(k, 5, cW);
    SETARG(k, 6, cCo);
    SETARG(k, 7, total);
    SETARG(k, 8, cact);
    run1d(m, k, total, "convt_scatter");
}

/* Multi-head attention core on already projected tensors (decoder, 8
 * heads). qp/kp/vp hold P prompts of Nq/Nk rows of dint channels with
 * per-prompt strides sq/sk/sv (0 = shared by all prompts); the
 * recombined heads are written to o as P * Nq rows of dint. Scores,
 * softmax and weighting are fused: per query for the 8 prompt-token
 * keys, per work-group (online softmax) for the 5184 image-token keys. */
static void mha(struct sam_model *m, int P, int Nq, int Nk, int dint, cl_mem qp,
                long sq, cl_mem kp, long sk, cl_mem vp, long sv, cl_mem o)
{
    cl_int c = dint / DEC_HEADS, cNq = Nq, cNk = Nk, cd = dint;
    cl_int csq = (cl_int)sq, csk = (cl_int)sk, csv = (cl_int)sv;
    cl_int total = P * DEC_HEADS * Nq;
    cl_float scale = 1.0f / sqrtf((float)c);
    int many = Nk > 64;
    cl_kernel k = many ? m->k_attn_many : m->k_attn_few;

    if (many && c != 16)
        G_fatal_error("mha: unsupported shape");
    SETARG(k, 0, qp);
    SETARG(k, 1, csq);
    SETARG(k, 2, kp);
    SETARG(k, 3, csk);
    SETARG(k, 4, vp);
    SETARG(k, 5, csv);
    SETARG(k, 6, o);
    SETARG(k, 7, cNq);
    SETARG(k, 8, cNk);
    SETARG(k, 9, cd);
    SETARG(k, 10, c);
    SETARG(k, 11, scale);
    SETARG(k, 12, total);
    if (many) {
        size_t gs = (size_t)total * 256, ls = 256;

        run(m, k, 1, &gs, &ls, "attn_manykeys");
    }
    else
        run1d(m, k, total, "attn_fewkeys");
}

/* Model loading. */

/* RoPE table of a block as (cos, sin) pairs, from the complex freqs_cis
 * buffer the checkpoint stores (positions x head_dim / 2). */
static cl_mem load_rope(struct sam_model *m, const struct pth_file *pf, int i,
                        long npos)
{
    char name[160];
    const struct pth_tensor *t;
    float *h;
    cl_mem b;

    snprintf(name, sizeof(name), TRUNK ".blocks.%d.attn.freqs_cis", i);
    t = pth_require(pf, name, 2, (long[]){npos, m->hd / 2});
    h = pth_read_c64(pf, t);
    b = ocl_upload(m->ocl, h, (size_t)npos * m->hd * sizeof(float), name);
    G_free(h);
    return b;
}

static void load_encoder(struct sam_model *m, const struct pth_file *pf)
{
    const struct pth_tensor *t;
    const long T = SAM_GRID * SAM_GRID;
    long E, s1[1], s2[2];
    char name[160];
    int i, rope_win = -1, rope_glob = -1;
    float *pe, *pos;

    t = pth_require(pf, TRUNK ".pos_embed", 3, (long[]){1, -1, -1});
    E = t->shape[2];
    m->embed = (int)E;
    /* Pretraining grid plus the class token SAM 3 drops. */
    m->pt_side = (int)sqrt((double)(t->shape[1] - 1));
    if ((long)m->pt_side * m->pt_side + 1 != t->shape[1] ||
        SAM_GRID % m->pt_side != 0)
        G_fatal_error(_("Unsupported position embedding in <%s>"), pf->path);
    pe = pth_read_f32(pf, t);
    pos = G_malloc(T * E * sizeof(float));
    /* tile_abs_pos: the pretraining grid repeated over the 72 x 72 grid. */
    for (i = 0; i < T; i++) {
        int y = i / SAM_GRID % m->pt_side, x = i % SAM_GRID % m->pt_side;

        memcpy(pos + (size_t)i * E, pe + (1 + (size_t)y * m->pt_side + x) * E,
               E * sizeof(float));
    }
    m->pos =
        ocl_upload(m->ocl, pos, T * E * sizeof(float), "position embedding");
    G_free(pe);
    G_free(pos);

    for (m->depth = 0;; m->depth++) {
        snprintf(name, sizeof(name), TRUNK ".blocks.%d.norm1.weight", m->depth);
        if (!pth_find(pf, name))
            break;
    }
    t = pth_require(pf, TRUNK ".blocks.0.attn.freqs_cis", 2, NULL);
    m->hd = (int)t->shape[1] * 2;
    if (m->hd != 64 || E % m->hd != 0)
        G_fatal_error(_("Unsupported attention head width %d"), m->hd);
    m->heads = (int)(E / m->hd);
    t = pth_require(pf, TRUNK ".blocks.0.mlp.fc1.weight", 2, (long[]){-1, E});
    m->mlp = (int)t->shape[0];
    G_message(_("SAM 3 image encoder: width %d, depth %d, %d heads"), m->embed,
              m->depth, m->heads);

    m->pe_w = load_dev(m, pf, TRUNK ".patch_embed.proj.weight", 4,
                       (long[]){E, 3, SAM_PATCHSZ, SAM_PATCHSZ});
    m->lnpw = load_dev(m, pf, TRUNK ".ln_pre.weight", 1, (long[]){E});
    m->lnpb = load_dev(m, pf, TRUNK ".ln_pre.bias", 1, (long[]){E});

    m->blocks = G_calloc(m->depth, sizeof(struct vit_block));
    for (i = 0; i < m->depth; i++) {
        struct vit_block *b = &m->blocks[i];
        const char *p = TRUNK ".blocks";
        long npos;

        /* The block's RoPE table covers its attention extent: a window
         * (squared window side) or the whole grid. */
        snprintf(name, sizeof(name), "%s.%d.attn.freqs_cis", p, i);
        npos = pth_require(pf, name, 2, (long[]){-1, m->hd / 2})->shape[0];
        if (npos == T)
            b->window = 0;
        else {
            b->window = (int)sqrt((double)npos);
            if ((long)b->window * b->window != npos || b->window > SAM_GRID)
                G_fatal_error(_("Unsupported attention extent in <%s>"), name);
        }
        if (b->window && rope_win < 0) {
            rope_win = i;
            m->rope_win = load_rope(m, pf, i, npos);
        }
        else if (!b->window && rope_glob < 0) {
            rope_glob = i;
            m->rope_glob = load_rope(m, pf, i, npos);
        }
        else if (b->window && b->window != m->blocks[rope_win].window)
            G_fatal_error(_("Blocks with different window sizes in <%s>"),
                          pf->path);

        s1[0] = E;
        b->n1w = load_dev_fmt(m, pf, 1, s1, "%s.%d.norm1.weight", p, i);
        b->n1b = load_dev_fmt(m, pf, 1, s1, "%s.%d.norm1.bias", p, i);
        b->n2w = load_dev_fmt(m, pf, 1, s1, "%s.%d.norm2.weight", p, i);
        b->n2b = load_dev_fmt(m, pf, 1, s1, "%s.%d.norm2.bias", p, i);
        s2[0] = 3 * E;
        s2[1] = E;
        b->qkvw = load_dev_w_fmt(m, pf, 2, s2, "%s.%d.attn.qkv.weight", p, i);
        s1[0] = 3 * E;
        b->qkvb = load_dev_fmt(m, pf, 1, s1, "%s.%d.attn.qkv.bias", p, i);
        s2[0] = E;
        b->projw = load_dev_w_fmt(m, pf, 2, s2, "%s.%d.attn.proj.weight", p, i);
        s1[0] = E;
        b->projb = load_dev_fmt(m, pf, 1, s1, "%s.%d.attn.proj.bias", p, i);
        s2[0] = m->mlp;
        s2[1] = E;
        b->l1w = load_dev_w_fmt(m, pf, 2, s2, "%s.%d.mlp.fc1.weight", p, i);
        s1[0] = m->mlp;
        b->l1b = load_dev_fmt(m, pf, 1, s1, "%s.%d.mlp.fc1.bias", p, i);
        s2[0] = E;
        s2[1] = m->mlp;
        b->l2w = load_dev_w_fmt(m, pf, 2, s2, "%s.%d.mlp.fc2.weight", p, i);
        s1[0] = E;
        b->l2b = load_dev_fmt(m, pf, 1, s1, "%s.%d.mlp.fc2.bias", p, i);
    }
    if (rope_win < 0 || rope_glob < 0)
        G_fatal_error(_("<%s> lacks windowed or global attention blocks"),
                      pf->path);
}

/* The first three levels of a neck (prefix: its "convs" module); the
 * fourth, 0.5x level is dropped by the backbone (scalp = 1). */
void load_neck(struct sam_model *m, const struct pth_file *pf,
               const char *prefix, struct neck_level lv[3])
{
    const long E = m->embed;
    int i;

    for (i = 0; i < 3; i++) {
        struct neck_level *n = &lv[i];
        long cin = E;

        n->scale = 4 >> i;
        if (n->scale == 4) {
            n->d0w = load_dev_fmt(m, pf, 4, (long[]){E, E / 2, 2, 2},
                                  "%s.%d.dconv_2x2_0.weight", prefix, i);
            n->d0b = load_dev_fmt(m, pf, 1, (long[]){E / 2},
                                  "%s.%d.dconv_2x2_0.bias", prefix, i);
            n->d1w = load_dev_fmt(m, pf, 4, (long[]){E / 2, E / 4, 2, 2},
                                  "%s.%d.dconv_2x2_1.weight", prefix, i);
            n->d1b = load_dev_fmt(m, pf, 1, (long[]){E / 4},
                                  "%s.%d.dconv_2x2_1.bias", prefix, i);
            cin = E / 4;
        }
        else if (n->scale == 2) {
            n->d0w = load_dev_fmt(m, pf, 4, (long[]){E, E / 2, 2, 2},
                                  "%s.%d.dconv_2x2.weight", prefix, i);
            n->d0b = load_dev_fmt(m, pf, 1, (long[]){E / 2},
                                  "%s.%d.dconv_2x2.bias", prefix, i);
            cin = E / 2;
        }
        n->c1w = load_dev_fmt(m, pf, 4, (long[]){SAM_DIM, cin, 1, 1},
                              "%s.%d.conv_1x1.weight", prefix, i);
        n->c1b = load_dev_fmt(m, pf, 1, (long[]){SAM_DIM},
                              "%s.%d.conv_1x1.bias", prefix, i);
        n->c3w = load_dev_fmt(m, pf, 4, (long[]){SAM_DIM, SAM_DIM, 3, 3},
                              "%s.%d.conv_3x3.weight", prefix, i);
        n->c3b = load_dev_fmt(m, pf, 1, (long[]){SAM_DIM},
                              "%s.%d.conv_3x3.bias", prefix, i);
    }
}

void release_neck(struct neck_level lv[3])
{
    int j;
    size_t i;

    for (j = 0; j < 3; j++) {
        struct neck_level *n = &lv[j];
        cl_mem *nb[] = {&n->d0w, &n->d0b, &n->d1w, &n->d1b,
                        &n->c1w, &n->c1b, &n->c3w, &n->c3b};

        for (i = 0; i < sizeof(nb) / sizeof(nb[0]); i++)
            ocl_release(nb[i]);
    }
}

/* The neck of the point-prompt head (sam2_convs) and its decoder's
 * high-resolution projections. */
static void load_point_neck(struct sam_model *m, const struct pth_file *pf)
{
    load_neck(m, pf, NECK, m->neck);
    m->s0w = load_dev(m, pf, "tracker.sam_mask_decoder.conv_s0.weight", 4,
                      (long[]){SAM_DIM / 8, SAM_DIM, 1, 1});
    m->s0b = load_dev(m, pf, "tracker.sam_mask_decoder.conv_s0.bias", 1,
                      (long[]){SAM_DIM / 8});
    m->s1w = load_dev(m, pf, "tracker.sam_mask_decoder.conv_s1.weight", 4,
                      (long[]){SAM_DIM / 4, SAM_DIM, 1, 1});
    m->s1b = load_dev(m, pf, "tracker.sam_mask_decoder.conv_s1.bias", 1,
                      (long[]){SAM_DIM / 4});
    m->no_mem =
        load_dev(m, pf, "tracker.no_mem_embed", 3, (long[]){1, 1, SAM_DIM});
}

static void load_decoder(struct sam_model *m, const struct pth_file *pf)
{
    const long v256[1] = {SAM_DIM}, e1[2] = {1, SAM_DIM};
    const char *dec = "tracker.sam_mask_decoder";
    char name[160];
    int i, j;

    load_host_into(pf,
                   "tracker.sam_prompt_encoder.pe_layer.positional_encoding_"
                   "gaussian_matrix",
                   2, (long[]){2, SAM_DIM / 2}, m->gauss);
    for (i = 0; i < 4; i++) {
        snprintf(name, sizeof(name),
                 "tracker.sam_prompt_encoder.point_embeddings.%d.weight", i);
        load_host_into(pf, name, 2, e1, m->point_emb[i]);
    }
    load_host_into(pf, "tracker.sam_prompt_encoder.not_a_point_embed.weight", 2,
                   e1, m->not_a_point);
    load_host_into(pf, "tracker.sam_prompt_encoder.no_mask_embed.weight", 2, e1,
                   m->no_mask);
    load_host_into(pf, "tracker.sam_mask_decoder.obj_score_token.weight", 2, e1,
                   m->obj_token);
    load_host_into(pf, "tracker.sam_mask_decoder.iou_token.weight", 2, e1,
                   m->iou_token);
    load_host_into(pf, "tracker.sam_mask_decoder.mask_tokens.weight", 2,
                   (long[]){4, SAM_DIM}, m->mask_tokens);

    for (i = 0; i < 2; i++) {
        struct sam_dec_layer *L = &m->layers[i];
        const char *p = "tracker.sam_mask_decoder.transformer.layers";

        snprintf(name, sizeof(name), "%s.%d.self_attn", p, i);
        load_attn(m, pf, name, SAM_DIM, &L->self);
        snprintf(name, sizeof(name), "%s.%d.cross_attn_token_to_image", p, i);
        load_attn(m, pf, name, SAM_DIM / 2, &L->t2i);
        snprintf(name, sizeof(name), "%s.%d.cross_attn_image_to_token", p, i);
        load_attn(m, pf, name, SAM_DIM / 2, &L->i2t);
        L->n1w = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm1.weight", p, i);
        L->n1b = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm1.bias", p, i);
        L->n2w = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm2.weight", p, i);
        L->n2b = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm2.bias", p, i);
        L->n3w = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm3.weight", p, i);
        L->n3b = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm3.bias", p, i);
        L->n4w = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm4.weight", p, i);
        L->n4b = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm4.bias", p, i);
        L->m1w = load_dev_w_fmt(m, pf, 2, (long[]){2048, SAM_DIM},
                              "%s.%d.mlp.lin1.weight", p, i);
        L->m1b =
            load_dev_fmt(m, pf, 1, (long[]){2048}, "%s.%d.mlp.lin1.bias", p, i);
        L->m2w = load_dev_w_fmt(m, pf, 2, (long[]){SAM_DIM, 2048},
                              "%s.%d.mlp.lin2.weight", p, i);
        L->m2b = load_dev_fmt(m, pf, 1, v256, "%s.%d.mlp.lin2.bias", p, i);
    }
    load_attn(m, pf,
              "tracker.sam_mask_decoder.transformer.final_attn_token_to_image",
              SAM_DIM / 2, &m->final_t2i);
    m->nfw = load_dev_fmt(m, pf, 1, v256,
                          "%s.transformer.norm_final_attn.weight", dec);
    m->nfb = load_dev_fmt(m, pf, 1, v256, "%s.transformer.norm_final_attn.bias",
                          dec);

    m->up0w = load_dev_fmt(m, pf, 4, (long[]){SAM_DIM, 64, 2, 2},
                           "%s.output_upscaling.0.weight", dec);
    m->up0b =
        load_dev_fmt(m, pf, 1, (long[]){64}, "%s.output_upscaling.0.bias", dec);
    m->up1w = load_dev_fmt(m, pf, 1, (long[]){64},
                           "%s.output_upscaling.1.weight", dec);
    m->up1b =
        load_dev_fmt(m, pf, 1, (long[]){64}, "%s.output_upscaling.1.bias", dec);
    m->up3w = load_dev_fmt(m, pf, 4, (long[]){64, 32, 2, 2},
                           "%s.output_upscaling.3.weight", dec);
    m->up3b =
        load_dev_fmt(m, pf, 1, (long[]){32}, "%s.output_upscaling.3.bias", dec);

    for (i = 0; i < 4; i++) {
        for (j = 0; j < 3; j++) {
            long ws[2] = {j == 2 ? 32 : SAM_DIM, SAM_DIM};
            long bs[1] = {ws[0]};

            snprintf(name, sizeof(name),
                     "%s.output_hypernetworks_mlps.%d.layers.%d.weight", dec, i,
                     j);
            m->hyper_w[i][j] = load_host(pf, name, 2, ws);
            snprintf(name, sizeof(name),
                     "%s.output_hypernetworks_mlps.%d.layers.%d.bias", dec, i,
                     j);
            m->hyper_b[i][j] = load_host(pf, name, 1, bs);
        }
    }
    for (j = 0; j < 3; j++) {
        long ws[2] = {j == 2 ? 4 : SAM_DIM, SAM_DIM};
        long bs[1] = {ws[0]};

        snprintf(name, sizeof(name), "%s.iou_prediction_head.layers.%d.weight",
                 dec, j);
        m->iou_w[j] = load_host(pf, name, 2, ws);
        snprintf(name, sizeof(name), "%s.iou_prediction_head.layers.%d.bias",
                 dec, j);
        m->iou_b[j] = load_host(pf, name, 1, bs);
    }
}

/* PositionEmbeddingRandom encoding of normalized (x, y) in [0, 1]. */
static void pe_encode(const struct sam_model *m, float x, float y, float *out)
{
    int i;

    x = 2.0f * x - 1.0f;
    y = 2.0f * y - 1.0f;
    for (i = 0; i < SAM_DIM / 2; i++) {
        float v = x * m->gauss[i] + y * m->gauss[SAM_DIM / 2 + i];

        v *= 2.0f * (float)M_PI;
        out[i] = sinf(v);
        out[SAM_DIM / 2 + i] = cosf(v);
    }
}

/* Attention geometry of a ViT block: windows of side s (the whole grid
 * for global attention), nwx windows per side. */
struct geom {
    int s, nwx, nwin, N;
};

static struct geom block_geom(const struct vit_block *b)
{
    struct geom g;

    g.s = b->window ? b->window : SAM_GRID;
    g.nwx = (SAM_GRID + g.s - 1) / g.s;
    g.nwin = g.nwx * g.nwx;
    g.N = g.s * g.s;
    return g;
}

/* Heads whose score matrices fit at once in the score buffer. */
static int head_chunk(const struct sam_model *m, const struct geom *g)
{
    long hc = m->score_cap / ((long)g->nwin * g->N * g->N);

    return hc > m->heads ? m->heads : (int)hc;
}

#define GROW(v, n)                                                             \
    do {                                                                       \
        long n_ = (n);                                                         \
        if (n_ > (v))                                                          \
            (v) = n_;                                                          \
    } while (0)

static size_t work_bytes;

static cl_mem walloc(struct ocl_backend *ocl, size_t n, const char *what)
{
    work_bytes += n;
    return ocl_alloc(ocl, n, what);
}

static void alloc_buffers(struct sam_model *m)
{
    struct ocl_backend *ocl = m->ocl;
    const size_t f = sizeof(float);
    const long T = SAM_GRID * SAM_GRID, E = m->embed, P = m->batch;
    const long T4 = 16 * T; /* Tokens of the 4x neck level. */
    long rows = T, nscores = 0, nmlp, nneck;
    size_t cap;
    float *pe;
    int i, y, x;

    cap = ocl->max_alloc < ((size_t)512 << 20) ? ocl->max_alloc
                                               : ((size_t)512 << 20);
    m->score_cap = (long)(cap / f);
    for (i = 0; i < m->depth; i++) {
        const struct geom g = block_geom(&m->blocks[i]);
        int hc = head_chunk(m, &g);

        if (hc < 1)
            G_fatal_error(
                _("OpenCL device cannot hold the attention scores "
                  "of one head (%lu MiB)"),
                (unsigned long)((size_t)g.nwin * g.N * g.N * f >> 20));
        GROW(rows, (long)g.nwin * g.N);
        GROW(nscores, (long)hc * g.nwin * g.N * g.N);
    }
    nmlp = T * m->mlp > T * 3 * SAM_PATCHSZ * SAM_PATCHSZ
               ? T * m->mlp
               : T * 3 * SAM_PATCHSZ * SAM_PATCHSZ;
    /* Neck: transposed convolution GEMMs and upsampled maps, the largest
     * being the 288 x 288 x 256 level. */
    nneck = T * 2 * E;
    GROW(nneck, 4 * T * E);
    GROW(nneck, T4 * SAM_DIM);
    m->col_rows = (long)(((size_t)128 << 20) / (9 * SAM_DIM * f));
    if (m->col_rows > T4)
        m->col_rows = T4;

    m->img = walloc(ocl, 3L * SAM_IMG * SAM_IMG * f, "image");
    m->x = walloc(ocl, T * E * f, "tokens");
    m->xn = walloc(ocl, T * E * f, "normalized tokens");
    m->xw = walloc(ocl, rows * E * f, "window partition");
    m->qkv = walloc(ocl, rows * 3 * E * f, "qkv");
    m->scores = walloc(ocl, nscores * f, "attention scores");
    m->attn = walloc(ocl, rows * E * f, "attention output");
    m->tmp = walloc(ocl, rows * E * f, "projection");
    m->mlpbuf = walloc(ocl, nmlp * f, "mlp");
    m->nka = walloc(ocl, nneck * f, "neck a");
    m->nkb = walloc(ocl, nneck * f, "neck b");
    m->col = walloc(ocl, m->col_rows * 9 * SAM_DIM * f, "neck im2col");
    m->lowres =
        walloc(ocl, P * 3 * SAM_LOWRES * SAM_LOWRES * f, "low-res masks");
    m->h_lowres = G_malloc(P * 3 * SAM_LOWRES * SAM_LOWRES * f);
    m->sel = walloc(ocl, P * 3 * sizeof(cl_int), "mask selection");
    m->stats = walloc(ocl, P * 3 * 6 * sizeof(cl_int), "mask statistics");
    if (!m->point_head) {
        G_verbose_message(_("Allocated %lu MiB of work buffers"),
                          (unsigned long)(work_bytes >> 20));
        return;
    }

    m->emb = walloc(ocl, T * SAM_DIM * f, "embedding");
    m->feat_s0 = walloc(ocl, T4 * (SAM_DIM / 8) * f, "features s0");
    m->feat_s1 = walloc(ocl, 4 * T * (SAM_DIM / 4) * f, "features s1");

    m->src = walloc(ocl, T * SAM_DIM * f, "decoder src");
    m->srcpe = walloc(ocl, T * SAM_DIM * f, "decoder src+pe");
    m->kp0 = walloc(ocl, T * 128 * f, "decoder k0");
    m->vp0 = walloc(ocl, T * 128 * f, "decoder v0");
    m->qp0 = walloc(ocl, T * 128 * f, "decoder q0");

    /* Dense positional encoding of the 72 x 72 embedding grid. */
    pe = G_malloc(T * SAM_DIM * f);
    for (y = 0; y < SAM_GRID; y++)
        for (x = 0; x < SAM_GRID; x++)
            pe_encode(m, (x + 0.5f) / SAM_GRID, (y + 0.5f) / SAM_GRID,
                      pe + ((long)y * SAM_GRID + x) * SAM_DIM);
    m->pe_img = ocl_upload(ocl, pe, T * SAM_DIM * f, "decoder pe");
    G_free(pe);

    m->qpe = walloc(ocl, P * NTOK * SAM_DIM * f, "query pe");
    m->qry = walloc(ocl, P * NTOK * SAM_DIM * f, "queries");
    m->qin = walloc(ocl, P * NTOK * SAM_DIM * f, "queries+pe");
    m->sq = walloc(ocl, P * NTOK * SAM_DIM * f, "token q");
    m->sk = walloc(ocl, P * NTOK * SAM_DIM * f, "token k");
    m->sv = walloc(ocl, P * NTOK * SAM_DIM * f, "token v");
    m->so = walloc(ocl, P * NTOK * SAM_DIM * f, "token attention");
    m->sh = walloc(ocl, P * NTOK * 2048 * f, "token mlp");
    m->keys = walloc(ocl, P * T * SAM_DIM * f, "keys");
    m->kin = walloc(ocl, P * T * SAM_DIM * f, "keys+pe");
    m->bq = walloc(ocl, P * T * 128 * f, "image q");
    m->bk = walloc(ocl, P * T * 128 * f, "image k");
    m->bv = walloc(ocl, P * T * 128 * f, "image v");
    m->bo = walloc(ocl, P * T * 128 * f, "image attention");
    m->u1 = walloc(ocl, P * 4 * T * 64 * f, "upscaling 1");
    m->g2 = walloc(ocl, P * 4 * T * 128 * f, "upscaling 2 gemm");
    m->u2 = walloc(ocl, P * 16 * T * 32 * f, "upscaling 2");
    m->hyper = walloc(ocl, P * 3 * 32 * f, "hypernetwork");
    m->h_queries = G_malloc(P * NTOK * SAM_DIM * f);
    G_verbose_message(_("Allocated %lu MiB of work buffers"),
                      (unsigned long)(work_bytes >> 20));
}

void sam_load(struct sam_model *m, struct ocl_backend *ocl,
              const struct pth_file *pf, int batch, int point_head)
{
    memset(m, 0, sizeof(*m));
    m->ocl = ocl;
    m->batch = batch;
    m->point_head = point_head;

    m->k_gemm = ocl_kernel(ocl, "gemm");
    m->k_gemm_big = ocl_kernel(ocl, "gemm_big");
    m->k_gemm_cpu_t = ocl_kernel(ocl, "gemm_cpu_t");
    m->k_gemm_cpu_n = ocl_kernel(ocl, "gemm_cpu_n");
    if (ocl->fp16w) {
        m->k_gemm_h = ocl_kernel_h(ocl, "gemm");
        m->k_gemm_big_h = ocl_kernel_h(ocl, "gemm_big");
        m->k_gemm_cpu_t_h = ocl_kernel_h(ocl, "gemm_cpu_t");
        m->k_gemm_cpu_n_h = ocl_kernel_h(ocl, "gemm_cpu_n");
    }
    m->k_ln = ocl_kernel(ocl, "layernorm");
    m->k_softmax = ocl_kernel(ocl, "softmax_rows");
    m->k_attn_few = ocl_kernel(ocl, "attn_fewkeys");
    m->k_attn_many = ocl_kernel(ocl, "attn_manykeys");
    m->k_patch = ocl_kernel(ocl, "patch_im2col");
    m->k_add = ocl_kernel(ocl, "add_bcast");
    m->k_wpart = ocl_kernel(ocl, "window_partition");
    m->k_wunpart = ocl_kernel(ocl, "window_unpartition_add");
    m->k_rope = ocl_kernel(ocl, "rope_apply");
    m->k_im2col = ocl_kernel(ocl, "im2col3x3");
    m->k_convt = ocl_kernel(ocl, "convt_scatter");
    m->k_logits = ocl_kernel(ocl, "mask_logits");
    m->k_post = ocl_kernel(ocl, "mask_post");

    G_message(_("Loading SAM 3 checkpoint <%s>..."), pf->path);
    loaded_bytes = 0;
    load_encoder(m, pf);
    if (point_head) {
        load_point_neck(m, pf);
        load_decoder(m, pf);
    }
    G_verbose_message(_("Uploaded %lu MiB of weights"),
                      (unsigned long)(loaded_bytes >> 20));

    alloc_buffers(m);
}

/* Image encoder forward pass. */

static void rope_apply(struct sam_model *m, cl_mem table, long rows, int npos)
{
    cl_kernel k = m->k_rope;
    cl_int cE = m->embed, chd = m->hd, cnpos = npos;
    cl_int total = (cl_int)(rows * m->embed);

    SETARG(k, 0, m->qkv);
    SETARG(k, 1, cE);
    SETARG(k, 2, chd);
    SETARG(k, 3, cnpos);
    SETARG(k, 4, table);
    SETARG(k, 5, total);
    run1d(m, k, total, "rope_apply");
}

/* One ViT block, x updated in place. */
static void vit_block(struct sam_model *m, const struct vit_block *b)
{
    const struct geom g = block_geom(b);
    const int T = SAM_GRID * SAM_GRID, E = m->embed, hd = m->hd;
    const long rows = (long)g.nwin * g.N;
    const float scale = 1.0f / sqrtf((float)hd);
    const int hc = head_chunk(m, &g);
    cl_mem A = m->xn;
    int h0;

    layernorm(m, T, E, m->x, m->xn, b->n1w, b->n1b, ENC_EPS, ACT_NONE);
    if (b->window) {
        window_partition(m, m->xn, m->xw, SAM_GRID, E, g.s, g.nwx);
        A = m->xw;
    }
    linear(m, (int)rows, E, 3 * E, A, 0, b->qkvw, b->qkvb, m->qkv, 0, ACT_NONE,
           0);
    rope_apply(m, b->window ? m->rope_win : m->rope_glob, rows, g.N);

    for (h0 = 0; h0 < m->heads; h0 += hc) {
        int hcc = m->heads - h0 < hc ? m->heads - h0 : hc;
        long plane = (long)g.N * g.N, tstride = (long)g.N * 3 * E;

        gemm(m, g.N, g.N, hd, bmat(m->qkv, (long)h0 * hd, 3 * E, tstride, hd),
             bmat(m->qkv, E + (long)h0 * hd, 3 * E, tstride, hd), 1,
             bmat(m->scores, 0, g.N, hcc * plane, plane), g.nwin * hcc, hcc,
             NULL, NULL, 0, scale, ACT_NONE);
        softmax_rows(m, m->scores, (long)g.nwin * hcc * g.N, g.N);
        gemm(m, g.N, hd, g.N, bmat(m->scores, 0, g.N, hcc * plane, plane),
             bmat(m->qkv, 2L * E + (long)h0 * hd, 3 * E, tstride, hd), 0,
             bmat(m->attn, (long)h0 * hd, E, (long)g.N * E, hd), g.nwin * hcc,
             hcc, NULL, NULL, 0, 1.0f, ACT_NONE);
    }

    if (b->window) {
        linear(m, (int)rows, E, E, m->attn, 0, b->projw, b->projb, m->tmp, 0,
               ACT_NONE, 0);
        window_unpartition_add(m, m->x, m->tmp, SAM_GRID, E, g.s, g.nwx);
    }
    else
        linear(m, T, E, E, m->attn, 0, b->projw, b->projb, m->x, 0, ACT_NONE,
               1);

    layernorm(m, T, E, m->x, m->xn, b->n2w, b->n2b, ENC_EPS, ACT_NONE);
    linear(m, T, E, m->mlp, m->xn, 0, b->l1w, b->l1b, m->mlpbuf, 0, ACT_GELU,
           0);
    linear(m, T, m->mlp, E, m->mlpbuf, 0, b->l2w, b->l2b, m->x, 0, ACT_NONE, 1);
}

/* 3x3/pad-1 convolution of a side x side x SAM_DIM map X into Y, in bands
 * of col_rows tokens. */
void conv3x3(struct sam_model *m, cl_mem X, int side, cl_mem w, cl_mem b,
             cl_mem Y)
{
    const long T = (long)side * side;
    cl_kernel k = m->k_im2col;
    cl_int cH = side, cC = SAM_DIM;
    long t0;

    for (t0 = 0; t0 < T; t0 += m->col_rows) {
        long n = T - t0 < m->col_rows ? T - t0 : m->col_rows;
        cl_int ct0 = (cl_int)t0, total = (cl_int)(n * 9 * SAM_DIM);

        SETARG(k, 0, X);
        SETARG(k, 1, m->col);
        SETARG(k, 2, cH);
        SETARG(k, 3, cH);
        SETARG(k, 4, cC);
        SETARG(k, 5, ct0);
        SETARG(k, 6, total);
        run1d(m, k, total, "im2col3x3");
        linear(m, (int)n, 9 * SAM_DIM, SAM_DIM, m->col, 0, w, b, Y,
               t0 * SAM_DIM, ACT_NONE, 0);
    }
}

/* 2x2/stride-2 transposed convolution of a side x side x cin map X
 * (weight [cin, cout, 2, 2]) into a 2 side x 2 side x cout map Y, through
 * the GEMM buffer G. */
void convt2x2(struct sam_model *m, cl_mem X, int side, int cin, int cout,
              cl_mem w, cl_mem b, cl_mem G, cl_mem Y, int act)
{
    gemm(m, side * side, cout * 4, cin, mat(X, 0, cin), mat(w, 0, cout * 4), 0,
         mat(G, 0, cout * 4), 1, 1, NULL, NULL, 0, 1.0f, ACT_NONE);
    convt_scatter(m, G, Y, b, NULL, 1, side, side, cout, act);
}

/* One level of a SimpleFPN neck on the last ViT map (in x) into out, a
 * (4 >> i) * 72 square map of SAM_DIM channels; out may be nkb. */
void neck_level(struct sam_model *m, const struct neck_level *n, cl_mem out)
{
    const int G = SAM_GRID, E = m->embed;

    if (n->scale == 4) {
        convt2x2(m, m->x, G, E, E / 2, n->d0w, n->d0b, m->nka, m->nkb,
                 ACT_GELU);
        convt2x2(m, m->nkb, 2 * G, E / 2, E / 4, n->d1w, n->d1b, m->nka, m->nkb,
                 ACT_NONE);
        linear(m, 16 * G * G, E / 4, SAM_DIM, m->nkb, 0, n->c1w, n->c1b, m->nka,
               0, ACT_NONE, 0);
    }
    else if (n->scale == 2) {
        convt2x2(m, m->x, G, E, E / 2, n->d0w, n->d0b, m->nka, m->nkb,
                 ACT_NONE);
        linear(m, 4 * G * G, E / 2, SAM_DIM, m->nkb, 0, n->c1w, n->c1b, m->nka,
               0, ACT_NONE, 0);
    }
    else
        linear(m, G * G, E, SAM_DIM, m->x, 0, n->c1w, n->c1b, m->nka, 0,
               ACT_NONE, 0);
    conv3x3(m, m->nka, n->scale * G, n->c3w, n->c3b, out);
}

/* Neck of the point-prompt head and the decoder's high-resolution
 * projections: level 0 at 4x (288 x 288) feeds conv_s0, level 1 at 2x
 * conv_s1, level 2 at 1x is the image embedding, to which
 * SAM3InteractiveImagePredictor adds no_mem_embed. */
static void point_neck(struct sam_model *m)
{
    const int G = SAM_GRID;

    neck_level(m, &m->neck[0], m->nkb);
    linear(m, 16 * G * G, SAM_DIM, SAM_DIM / 8, m->nkb, 0, m->s0w, m->s0b,
           m->feat_s0, 0, ACT_NONE, 0);
    neck_level(m, &m->neck[1], m->nkb);
    linear(m, 4 * G * G, SAM_DIM, SAM_DIM / 4, m->nkb, 0, m->s1w, m->s1b,
           m->feat_s1, 0, ACT_NONE, 0);
    neck_level(m, &m->neck[2], m->emb);
    add_bcast(m, m->emb, m->emb, m->no_mem, (long)G * G * SAM_DIM, SAM_DIM);
}

void sam_encode_trunk(struct sam_model *m, const float *img)
{
    const long T = SAM_GRID * SAM_GRID;
    const int E = m->embed, K = 3 * SAM_PATCHSZ * SAM_PATCHSZ;
    int i;

    ocl_write(m->ocl, m->img, 0, img, 3L * SAM_IMG * SAM_IMG * sizeof(float));

    /* Patch embedding (no bias), tiled position embedding, ln_pre. */
    {
        cl_kernel k = m->k_patch;
        cl_int side = SAM_IMG, grid = SAM_GRID, ps = SAM_PATCHSZ;

        SETARG(k, 0, m->img);
        SETARG(k, 1, m->mlpbuf);
        SETARG(k, 2, side);
        SETARG(k, 3, grid);
        SETARG(k, 4, ps);
        run1d(m, k, (size_t)T * K, "patch_im2col");
        linear(m, (int)T, K, E, m->mlpbuf, 0, m->pe_w, NULL, m->x, 0, ACT_NONE,
               0);
        add_bcast(m, m->x, m->x, m->pos, T * E, T * E);
        layernorm(m, T, E, m->x, m->x, m->lnpw, m->lnpb, ENC_EPS, ACT_NONE);
    }

    for (i = 0; i < m->depth; i++) {
        /* Report the blocks the device has finished, not the ones
         * queued: enqueueing all of them takes milliseconds. */
        clFinish(m->ocl->queue);
        progress_part(i, m->depth);
        vit_block(m, &m->blocks[i]);
    }
}

void sam_encode(struct sam_model *m, const float *img)
{
    const long T = SAM_GRID * SAM_GRID;

    sam_encode_trunk(m, img);
    point_neck(m);

    /* Prompt-independent decoder inputs. Without a mask prompt the
     * dense embedding is no_mask_embed broadcast over the grid. */
    {
        cl_mem nm = ocl_upload(m->ocl, m->no_mask, SAM_DIM * sizeof(float),
                               "no-mask embedding");
        const struct sam_dec_layer *L0 = &m->layers[0];

        add_bcast(m, m->src, m->emb, nm, T * SAM_DIM, SAM_DIM);
        add_bcast(m, m->srcpe, m->src, m->pe_img, T * SAM_DIM, T * SAM_DIM);
        linear(m, (int)T, SAM_DIM, 128, m->srcpe, 0, L0->t2i.kw, L0->t2i.kb,
               m->kp0, 0, ACT_NONE, 0);
        linear(m, (int)T, SAM_DIM, 128, m->src, 0, L0->t2i.vw, L0->t2i.vb,
               m->vp0, 0, ACT_NONE, 0);
        linear(m, (int)T, SAM_DIM, 128, m->srcpe, 0, L0->i2t.qw, L0->i2t.qb,
               m->qp0, 0, ACT_NONE, 0);
        clFinish(m->ocl->queue);
        clReleaseMemObject(nm);
    }
}

void sam_read_features(struct sam_model *m, float *emb, float *s0, float *s1)
{
    const size_t f = sizeof(float), T = (size_t)SAM_GRID * SAM_GRID;

    ocl_read(m->ocl, m->emb, 0, emb, T * SAM_DIM * f);
    ocl_read(m->ocl, m->feat_s0, 0, s0, 16 * T * (SAM_DIM / 8) * f);
    ocl_read(m->ocl, m->feat_s1, 0, s1, 4 * T * (SAM_DIM / 4) * f);
}

/* Mask decoder forward pass. */

/* y = W x + b for a PyTorch [nout, nin] weight, optional ReLU. */
void host_linear(const float *W, const float *b, const float *x, int nin,
                 int nout, int relu, float *y)
{
    int o, i;

    for (o = 0; o < nout; o++) {
        const float *w = W + (long)o * nin;
        double acc = b[o];

        for (i = 0; i < nin; i++)
            acc += (double)w[i] * x[i];
        y[o] = relu && acc < 0.0 ? 0.0f : (float)acc;
    }
}

static void host_mlp3(float *const W[3], float *const B[3], const float *x,
                      int nout, float *y)
{
    float h1[SAM_DIM], h2[SAM_DIM];

    host_linear(W[0], B[0], x, SAM_DIM, SAM_DIM, 1, h1);
    host_linear(W[1], B[1], h1, SAM_DIM, SAM_DIM, 1, h2);
    host_linear(W[2], B[2], h2, SAM_DIM, nout, 0, y);
}

/* Token -> image cross attention on per-prompt keys, residual into the
 * queries (layers >= 1 and the final attention). */
static void t2i_attention(struct sam_model *m, int P,
                          const struct sam_attn_w *a)
{
    const long T = SAM_GRID * SAM_GRID, n = (long)P * NTOK * SAM_DIM;

    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    add_bcast(m, m->kin, m->keys, m->pe_img, P * T * SAM_DIM, T * SAM_DIM);
    linear(m, P * NTOK, SAM_DIM, 128, m->qin, 0, a->qw, a->qb, m->sq, 0,
           ACT_NONE, 0);
    linear(m, P * T, SAM_DIM, 128, m->kin, 0, a->kw, a->kb, m->bk, 0, ACT_NONE,
           0);
    linear(m, P * T, SAM_DIM, 128, m->keys, 0, a->vw, a->vb, m->bv, 0, ACT_NONE,
           0);
    mha(m, P, NTOK, T, 128, m->sq, NTOK * 128, m->bk, T * 128, m->bv, T * 128,
        m->so);
    linear(m, P * NTOK, 128, SAM_DIM, m->so, 0, a->ow, a->ob, m->qry, 0,
           ACT_NONE, 1);
}

static void token_mlp(struct sam_model *m, int P, const struct sam_dec_layer *L)
{
    linear(m, P * NTOK, SAM_DIM, 2048, m->qry, 0, L->m1w, L->m1b, m->sh, 0,
           ACT_RELU, 0);
    linear(m, P * NTOK, 2048, SAM_DIM, m->sh, 0, L->m2w, L->m2b, m->qry, 0,
           ACT_NONE, 1);
}

void sam_decode(struct sam_model *m, int P, const float *points, float *iou)
{
    const long T = SAM_GRID * SAM_GRID, n = (long)P * NTOK * SAM_DIM;
    const struct sam_dec_layer *L0 = &m->layers[0], *L1 = &m->layers[1];
    float *tok = m->h_queries, *hyper;
    int p, i;

    if (P > m->batch)
        G_fatal_error("sam_decode: batch overflow");

    /* Prompt tokens: object score token, IoU token, 4 mask tokens, the
     * positive point and the padding point SAM appends when no box is
     * given. */
    for (p = 0; p < P; p++) {
        float *t = tok + (long)p * NTOK * SAM_DIM;

        memcpy(t, m->obj_token, SAM_DIM * sizeof(float));
        memcpy(t + SAM_DIM, m->iou_token, SAM_DIM * sizeof(float));
        memcpy(t + 2 * SAM_DIM, m->mask_tokens, 4 * SAM_DIM * sizeof(float));
        pe_encode(m, (points[2 * p] + 0.5f) / SAM_IMG,
                  (points[2 * p + 1] + 0.5f) / SAM_IMG, t + 6 * SAM_DIM);
        for (i = 0; i < SAM_DIM; i++) {
            t[6 * SAM_DIM + i] += m->point_emb[1][i];
            t[7 * SAM_DIM + i] = m->not_a_point[i];
        }
    }
    ocl_write(m->ocl, m->qpe, 0, tok, n * sizeof(float));
    ocl_check(clEnqueueCopyBuffer(m->ocl->queue, m->qpe, m->qry, 0, 0,
                                  n * sizeof(float), 0, NULL, NULL),
              "clEnqueueCopyBuffer");

    /* Layer 0: self attention without positional encoding and without
     * residual (skip_first_layer_pe). */
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qry, 0, L0->self.qw, L0->self.qb,
           m->sq, 0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qry, 0, L0->self.kw, L0->self.kb,
           m->sk, 0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qry, 0, L0->self.vw, L0->self.vb,
           m->sv, 0, ACT_NONE, 0);
    mha(m, P, NTOK, NTOK, SAM_DIM, m->sq, NTOK * SAM_DIM, m->sk, NTOK * SAM_DIM,
        m->sv, NTOK * SAM_DIM, m->so);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->so, 0, L0->self.ow, L0->self.ob,
           m->qry, 0, ACT_NONE, 0);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L0->n1w, L0->n1b, DEC_EPS,
              ACT_NONE);

    /* Layer 0 token -> image: keys are still the shared image embedding,
     * so the cached projections serve every prompt (stride 0). */
    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    linear(m, P * NTOK, SAM_DIM, 128, m->qin, 0, L0->t2i.qw, L0->t2i.qb, m->sq,
           0, ACT_NONE, 0);
    mha(m, P, NTOK, T, 128, m->sq, NTOK * 128, m->kp0, 0, m->vp0, 0, m->so);
    linear(m, P * NTOK, 128, SAM_DIM, m->so, 0, L0->t2i.ow, L0->t2i.ob, m->qry,
           0, ACT_NONE, 1);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L0->n2w, L0->n2b, DEC_EPS,
              ACT_NONE);
    token_mlp(m, P, L0);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L0->n3w, L0->n3b, DEC_EPS,
              ACT_NONE);

    /* Layer 0 image -> token, queries from the shared embedding. */
    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    linear(m, P * NTOK, SAM_DIM, 128, m->qin, 0, L0->i2t.kw, L0->i2t.kb, m->sk,
           0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, 128, m->qry, 0, L0->i2t.vw, L0->i2t.vb, m->sv,
           0, ACT_NONE, 0);
    mha(m, P, T, NTOK, 128, m->qp0, 0, m->sk, NTOK * 128, m->sv, NTOK * 128,
        m->bo);
    linear(m, P * T, 128, SAM_DIM, m->bo, 0, L0->i2t.ow, L0->i2t.ob, m->keys, 0,
           ACT_NONE, 0);
    add_bcast(m, m->keys, m->keys, m->src, P * T * SAM_DIM, T * SAM_DIM);
    layernorm(m, P * T, SAM_DIM, m->keys, m->keys, L0->n4w, L0->n4b, DEC_EPS,
              ACT_NONE);

    /* Layer 1. */
    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qin, 0, L1->self.qw, L1->self.qb,
           m->sq, 0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qin, 0, L1->self.kw, L1->self.kb,
           m->sk, 0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qry, 0, L1->self.vw, L1->self.vb,
           m->sv, 0, ACT_NONE, 0);
    mha(m, P, NTOK, NTOK, SAM_DIM, m->sq, NTOK * SAM_DIM, m->sk, NTOK * SAM_DIM,
        m->sv, NTOK * SAM_DIM, m->so);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->so, 0, L1->self.ow, L1->self.ob,
           m->qry, 0, ACT_NONE, 1);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L1->n1w, L1->n1b, DEC_EPS,
              ACT_NONE);
    t2i_attention(m, P, &L1->t2i);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L1->n2w, L1->n2b, DEC_EPS,
              ACT_NONE);
    token_mlp(m, P, L1);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L1->n3w, L1->n3b, DEC_EPS,
              ACT_NONE);

    /* Layer 1 image -> token (kin = keys + pe from t2i_attention). */
    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    linear(m, P * T, SAM_DIM, 128, m->kin, 0, L1->i2t.qw, L1->i2t.qb, m->bq, 0,
           ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, 128, m->qin, 0, L1->i2t.kw, L1->i2t.kb, m->sk,
           0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, 128, m->qry, 0, L1->i2t.vw, L1->i2t.vb, m->sv,
           0, ACT_NONE, 0);
    mha(m, P, T, NTOK, 128, m->bq, T * 128, m->sk, NTOK * 128, m->sv,
        NTOK * 128, m->bo);
    linear(m, P * T, 128, SAM_DIM, m->bo, 0, L1->i2t.ow, L1->i2t.ob, m->keys, 0,
           ACT_NONE, 1);
    layernorm(m, P * T, SAM_DIM, m->keys, m->keys, L1->n4w, L1->n4b, DEC_EPS,
              ACT_NONE);

    /* Final token -> image attention. */
    t2i_attention(m, P, &m->final_t2i);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, m->nfw, m->nfb, DEC_EPS,
              ACT_NONE);

    /* Output upscaling with the high-resolution features: ConvT 256->64
     * + feat_s1, LayerNorm2d, GELU, ConvT 64->32 + feat_s0, GELU. The kin
     * buffer is free again and holds the first GEMM. */
    gemm(m, P * T, 256, SAM_DIM, mat(m->keys, 0, SAM_DIM), mat(m->up0w, 0, 256),
         0, mat(m->kin, 0, 256), 1, 1, NULL, NULL, 0, 1.0f, ACT_NONE);
    convt_scatter(m, m->kin, m->u1, m->up0b, m->feat_s1, P, SAM_GRID, SAM_GRID,
                  64, ACT_NONE);
    layernorm(m, P * 4 * T, 64, m->u1, m->u1, m->up1w, m->up1b, LN2D_EPS,
              ACT_GELU);
    gemm(m, P * 4 * T, 128, 64, mat(m->u1, 0, 64), mat(m->up3w, 0, 128), 0,
         mat(m->g2, 0, 128), 1, 1, NULL, NULL, 0, 1.0f, ACT_NONE);
    convt_scatter(m, m->g2, m->u2, m->up3b, m->feat_s0, P, 2 * SAM_GRID,
                  2 * SAM_GRID, 32, ACT_GELU);

    /* Hypernetwork MLPs and IoU head are tiny: run them on the host. The
     * multimask outputs are mask tokens 1..3 (rows 3..5), the IoU head
     * reads row 1 and ends in a sigmoid (iou_prediction_use_sigmoid). */
    ocl_read(m->ocl, m->qry, 0, tok, n * sizeof(float));
    hyper = G_malloc((size_t)P * 96 * sizeof(float));
#pragma omp parallel for private(i)
    for (p = 0; p < P; p++) {
        const float *t = tok + (long)p * NTOK * SAM_DIM;
        float out[4];

        for (i = 1; i <= 3; i++)
            host_mlp3(m->hyper_w[i], m->hyper_b[i], t + (2 + i) * SAM_DIM, 32,
                      hyper + (long)p * 96 + (i - 1) * 32);
        host_mlp3(m->iou_w, m->iou_b, t + SAM_DIM, 4, out);
        for (i = 0; i < 3; i++)
            iou[p * 3 + i] = 1.0f / (1.0f + expf(-out[1 + i]));
    }
    ocl_write(m->ocl, m->hyper, 0, hyper, (size_t)P * 96 * sizeof(float));
    G_free(hyper);

    {
        cl_kernel k = m->k_logits;
        cl_int npix = SAM_LOWRES * SAM_LOWRES;
        cl_int total = P * npix;

        SETARG(k, 0, m->u2);
        SETARG(k, 1, m->hyper);
        SETARG(k, 2, m->lowres);
        SETARG(k, 3, npix);
        SETARG(k, 4, total);
        run1d(m, k, total, "mask_logits");
    }
}

/* SAM3InteractiveImagePredictor's max_hole_area: background components
 * (logit <= 0, 8-connected as skimage.measure.label labels them) of at
 * most max_area pixels in an L x L logit map are filled with logit 10.
 * Returns 1 if the map changed. */
static int fill_holes(float *v, int L, int max_area, int *lab, int *stack)
{
    const int n = L * L;
    int i, changed = 0;

    memset(lab, 0, n * sizeof(int));
    for (i = 0; i < n; i++) {
        int sp = 0, j;

        if (lab[i] || v[i] > 0.0f)
            continue;
        lab[i] = 1;
        stack[sp++] = i;
        /* Breadth-first: the queue doubles as the component's pixels. */
        for (j = 0; j < sp; j++) {
            int c = stack[j], cx = c % L, cy = c / L, dx, dy;

            for (dy = -1; dy <= 1; dy++)
                for (dx = -1; dx <= 1; dx++) {
                    int nx = cx + dx, ny = cy + dy, k;

                    if (nx < 0 || ny < 0 || nx >= L || ny >= L)
                        continue;
                    k = ny * L + nx;
                    if (!lab[k] && v[k] <= 0.0f) {
                        lab[k] = 1;
                        stack[sp++] = k;
                    }
                }
        }
        if (sp <= max_area) {
            for (j = 0; j < sp; j++)
                v[stack[j]] = 10.0f;
            changed = 1;
        }
    }
    return changed;
}

void sam_postprocess(struct sam_model *m, int n, const int *sel, int out_h,
                     int out_w, float offset, int fill, int *stats)
{
    const int L = SAM_LOWRES;
    const size_t plane = (size_t)L * L * sizeof(float);
    cl_kernel k = m->k_post;
    size_t need = (size_t)n * out_h * out_w, g[3], l[3] = {16, 16, 1};
    cl_int cL = L, coh = out_h, cow = out_w;
    cl_float coff = offset;
    char *changed;
    int i;

    if (n <= 0)
        return;

    /* Hole filling on the host, for the selected masks only. */
    changed = G_calloc(n, 1);
    if (!fill)
        goto resize;
    for (i = 0; i < n; i++)
        ocl_check(clEnqueueReadBuffer(
                      m->ocl->queue, m->lowres, CL_FALSE, sel[i] * plane, plane,
                      m->h_lowres + (size_t)i * L * L, 0, NULL, NULL),
                  "clEnqueueReadBuffer");
    clFinish(m->ocl->queue);
#pragma omp parallel
    {
        int *lab = G_malloc((size_t)L * L * sizeof(int));
        int *stack = G_malloc((size_t)L * L * sizeof(int));

#pragma omp for schedule(dynamic)
        for (i = 0; i < n; i++)
            changed[i] = (char)fill_holes(m->h_lowres + (size_t)i * L * L, L,
                                          SAM_HOLE, lab, stack);
        G_free(lab);
        G_free(stack);
    }
    for (i = 0; i < n; i++)
        if (changed[i])
            ocl_write(m->ocl, m->lowres, sel[i] * plane,
                      m->h_lowres + (size_t)i * L * L, plane);
resize:
    G_free(changed);
    if (need > m->mask_cap) {
        ocl_release(&m->mask);
        m->mask = ocl_alloc(m->ocl, need, "postprocessed masks");
        m->mask_cap = need;
    }
    for (i = 0; i < n; i++) {
        stats[6 * i + 0] = 0;
        stats[6 * i + 1] = 0;
        stats[6 * i + 2] = 0x7fffffff;
        stats[6 * i + 3] = 0x7fffffff;
        stats[6 * i + 4] = -1;
        stats[6 * i + 5] = -1;
    }
    ocl_write(m->ocl, m->sel, 0, sel, n * sizeof(cl_int));
    ocl_write(m->ocl, m->stats, 0, stats, 6 * n * sizeof(cl_int));

    SETARG(k, 0, m->lowres);
    SETARG(k, 1, m->sel);
    SETARG(k, 2, cL);
    SETARG(k, 3, coh);
    SETARG(k, 4, cow);
    SETARG(k, 5, coff);
    SETARG(k, 6, m->mask);
    SETARG(k, 7, m->stats);
    g[0] = (size_t)(out_w + 15) / 16 * 16;
    g[1] = (size_t)(out_h + 15) / 16 * 16;
    g[2] = n;
    run(m, k, 3, g, l, "mask_post");
    ocl_read(m->ocl, m->stats, 0, stats, 6 * n * sizeof(cl_int));
}

void sam_read_mask(struct sam_model *m, int k, int out_h, int out_w, int x0,
                   int y0, int x1, int y1, unsigned char *dst)
{
    size_t origin[3] = {(size_t)x0, (size_t)y0, (size_t)k};
    size_t region[3] = {(size_t)(x1 - x0 + 1), (size_t)(y1 - y0 + 1), 1};

    ocl_check(clEnqueueReadBufferRect(m->ocl->queue, m->mask, CL_TRUE, origin,
                                      (size_t[]){0, 0, 0}, region, out_w,
                                      (size_t)out_w * out_h, region[0], 0, dst,
                                      0, NULL, NULL),
              "clEnqueueReadBufferRect");
}

void sam_free(struct sam_model *m)
{
    int p;

    for (p = 0; p < nprof; p++) {
        if (prof[p].flops > 0.0 && prof[p].wall_ms > 0.0)
            G_message("kernel %-24s %8.1f ms (wall %8.1f ms) %7ld calls "
                      "%8.1f GFLOP/s",
                      prof[p].what, prof[p].ms, prof[p].wall_ms, prof[p].calls,
                      prof[p].flops / (prof[p].wall_ms * 1e6));
        else
            G_message("kernel %-24s %8.1f ms (wall %8.1f ms) %7ld calls",
                      prof[p].what, prof[p].ms, prof[p].wall_ms,
                      prof[p].calls);
    }

    cl_mem *bufs[] = {
        &m->pe_w,      &m->pos,   &m->lnpw,    &m->lnpb,    &m->rope_win,
        &m->rope_glob, &m->s0w,   &m->s0b,     &m->s1w,     &m->s1b,
        &m->no_mem,    &m->nfw,   &m->nfb,     &m->up0w,    &m->up0b,
        &m->up1w,      &m->up1b,  &m->up3w,    &m->up3b,    &m->img,
        &m->x,         &m->xn,    &m->xw,      &m->qkv,     &m->scores,
        &m->attn,      &m->tmp,   &m->mlpbuf,  &m->nka,     &m->nkb,
        &m->col,       &m->emb,   &m->feat_s0, &m->feat_s1, &m->src,
        &m->pe_img,    &m->srcpe, &m->kp0,     &m->vp0,     &m->qp0,
        &m->qpe,       &m->qry,   &m->qin,     &m->sq,      &m->sk,
        &m->sv,        &m->so,    &m->sh,      &m->keys,    &m->kin,
        &m->bq,        &m->bk,    &m->bv,      &m->bo,      &m->u1,
        &m->g2,        &m->u2,    &m->hyper,   &m->lowres,  &m->mask,
        &m->sel,       &m->stats};
    cl_kernel kernels[] = {m->k_gemm,      m->k_gemm_big, m->k_attn_few,
                           m->k_attn_many, m->k_ln,       m->k_softmax,
                           m->k_patch,     m->k_add,      m->k_wpart,
                           m->k_wunpart,   m->k_rope,     m->k_im2col,
                           m->k_convt,     m->k_logits,   m->k_post,
                           m->k_gemm_cpu_t, m->k_gemm_cpu_n, m->k_gemm_h,
                           m->k_gemm_big_h, m->k_gemm_cpu_t_h,
                           m->k_gemm_cpu_n_h};
    size_t i;
    int j;

    for (i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
        ocl_release(bufs[i]);
    release_neck(m->neck);
    for (j = 0; j < m->depth; j++) {
        struct vit_block *b = &m->blocks[j];
        cl_mem *bb[] = {&b->n1w,   &b->n1b,   &b->qkvw, &b->qkvb,
                        &b->projw, &b->projb, &b->n2w,  &b->n2b,
                        &b->l1w,   &b->l1b,   &b->l2w,  &b->l2b};

        for (i = 0; i < sizeof(bb) / sizeof(bb[0]); i++)
            ocl_release(bb[i]);
    }
    G_free(m->blocks);
    for (j = 0; j < 2; j++) {
        struct sam_dec_layer *L = &m->layers[j];
        cl_mem *lb[] = {&L->n1w, &L->n1b, &L->n2w, &L->n2b, &L->n3w, &L->n3b,
                        &L->n4w, &L->n4b, &L->m1w, &L->m1b, &L->m2w, &L->m2b};

        release_attn(&L->self);
        release_attn(&L->t2i);
        release_attn(&L->i2t);
        for (i = 0; i < sizeof(lb) / sizeof(lb[0]); i++)
            ocl_release(lb[i]);
    }
    release_attn(&m->final_t2i);
    for (j = 0; j < 4; j++) {
        for (i = 0; i < 3; i++) {
            G_free(m->hyper_w[j][i]);
            G_free(m->hyper_b[j][i]);
        }
    }
    for (i = 0; i < 3; i++) {
        G_free(m->iou_w[i]);
        G_free(m->iou_b[i]);
    }
    for (i = 0; i < sizeof(kernels) / sizeof(kernels[0]); i++)
        if (kernels[i])
            clReleaseKernel(kernels[i]);
    G_free(m->h_queries);
    G_free(m->h_lowres);
    memset(m, 0, sizeof(*m));
}
