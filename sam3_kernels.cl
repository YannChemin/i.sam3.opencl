/****************************************************************************
 *
 * MODULE:       i.sam3.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      OpenCL kernels for the SAM 3 ViT image encoder, the neck of
 *               its point-prompt head and the mask decoder. Embedded into
 *               the module at build time (see Makefile).
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

/* Tensor layout convention: activations are token-major (row = spatial
 * position or prompt token, column = channel), which makes every
 * nn.Linear and 1x1 convolution a plain row-major GEMM against the
 * PyTorch [out, in] weight matrix.
 *
 * Element-wise kernels index with 32-bit integers: 64-bit division is
 * emulated on GPUs and was measured to dominate their run time. The
 * host keeps every tensor below 2^31 elements. */

#define DEC_HEADS 8

#define ACT_NONE 0
#define ACT_GELU 1
#define ACT_RELU 2

float gelu_erf(float x)
{
    return 0.5f * x * (1.0f + erf(x * 0.70710678118654752f));
}

float apply_act(float v, int act)
{
    if (act == ACT_GELU)
        return gelu_erf(v);
    if (act == ACT_RELU)
        return fmax(v, 0.0f);
    return v;
}

/* Batched GEMM: C = act(alpha * A * op(B) + bias) + R.
 *
 * A is M x K row-major (lda). With transb != 0, B is stored N x K (the
 * PyTorch Linear weight layout, B(k, n) = B[n * ldb + k]); otherwise B
 * is K x N (B(k, n) = B[k * ldb + n]). Batch index b = get_group_id(2)
 * is split into (b / nb1, b % nb1), each part with its own element
 * stride per operand; a stride of 0 shares an operand across that batch
 * axis. Element offsets aoff/boff/coff/roff select sub-matrices of the
 * buffers. R, when used, has C's layout and may alias C for in-place
 * residual updates.
 *
 * Generic version: 64 x 64 output tile per work-group of 16 x 16 items,
 * each item computing a 4 x 4 register block. */
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void
gemm(const int M, const int N, const int K, __global const float *A,
     const long aoff, const int lda, const long sa0, const long sa1,
     __global const float *B, const long boff, const int ldb, const long sb0,
     const long sb1, const int transb, __global float *C, const long coff,
     const int ldc, const long sc0, const long sc1, const int nb1,
     __global const float *bias, __global const float *R, const long roff,
     const float alpha, const int act)
{
    __local float As[16][65];
    __local float Bs[16][65];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int tid = ty * 16 + tx;
    const int n0 = get_group_id(0) * 64, m0 = get_group_id(1) * 64;
    const int b = get_group_id(2);
    const long b0 = b / nb1, b1 = b % nb1;
    float acc[4][4];
    int i, j, k, l;

    A += aoff + b0 * sa0 + b1 * sa1;
    B += boff + b0 * sb0 + b1 * sb1;
    C += coff + b0 * sc0 + b1 * sc1;
    if (R)
        R += roff + b0 * sc0 + b1 * sc1;

    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (l = 0; l < 4; l++) {
            const int idx = tid + l * 256;
            const int r = idx >> 4, c = idx & 15;
            const int gm = m0 + r, gk = k0 + c;

            As[c][r] = (gm < M && gk < K) ? A[(long)gm * lda + gk] : 0.0f;
            if (transb) {
                const int gn = n0 + r;

                Bs[c][r] = (gn < N && gk < K) ? B[(long)gn * ldb + gk] : 0.0f;
            }
            else {
                const int kk = idx >> 6, nn = idx & 63;
                const int gk2 = k0 + kk, gn = n0 + nn;

                Bs[kk][nn] =
                    (gn < N && gk2 < K) ? B[(long)gk2 * ldb + gn] : 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (k = 0; k < 16; k++) {
            float a[4], bb[4];

            for (i = 0; i < 4; i++)
                a[i] = As[k][ty + 16 * i];
            for (j = 0; j < 4; j++)
                bb[j] = Bs[k][tx + 16 * j];
            for (i = 0; i < 4; i++)
                for (j = 0; j < 4; j++)
                    acc[i][j] = mad(a[i], bb[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (i = 0; i < 4; i++) {
        const int m = m0 + ty + 16 * i;

        if (m >= M)
            continue;
        for (j = 0; j < 4; j++) {
            const int n = n0 + tx + 16 * j;
            float v;

            if (n >= N)
                continue;
            v = alpha * acc[i][j];
            if (bias)
                v += bias[n];
            v = apply_act(v, act);
            if (R)
                v += R[(long)m * ldc + n];
            C[(long)m * ldc + n] = v;
        }
    }
}

/* Epilogue of gemm_big: one row of an 8 x 8 register block. */
void store_row8(__global float *C, __global const float *R,
                __global const float *bias, const int m, const int n,
                const int M, const int N, const int ldc, const float alpha,
                const int act, const float4 lo, const float4 hi)
{
    const float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};

    if (m >= M)
        return;
    for (int j = 0; j < 8; j++) {
        float x;

        if (n + j >= N)
            return;
        x = alpha * v[j];
        if (bias)
            x += bias[n + j];
        x = apply_act(x, act);
        if (R)
            x += R[(long)m * ldc + n + j];
        C[(long)m * ldc + n + j] = x;
    }
}

/* Rank-1 update of the 8 x 8 block row r from A column value s. */
#define ROW_FMA(r, s)                                                          \
    c##r##0 = mad((float4)(s), b0, c##r##0);                                   \
    c##r##1 = mad((float4)(s), b1, c##r##1)

/* Same contract as gemm, for large, 16-byte aligned operands: 128 x 128
 * output tile per 16 x 16 work-group, 8 x 8 register block per item
 * held in explicit float4 accumulators (private arrays of that size end
 * up in scratch memory with Mesa, costing a factor of 70), float4 loads.
 * Requires K, lda, ldb and all A/B offsets and batch strides to be
 * multiples of 4 (checked by the host). */
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void
gemm_big(const int M, const int N, const int K, __global const float *A,
         const long aoff, const int lda, const long sa0, const long sa1,
         __global const float *B, const long boff, const int ldb,
         const long sb0, const long sb1, const int transb, __global float *C,
         const long coff, const int ldc, const long sc0, const long sc1,
         const int nb1, __global const float *bias, __global const float *R,
         const long roff, const float alpha, const int act)
{
    __local float As[16][132];
    __local float Bs[16][132];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int tid = ty * 16 + tx;
    const int n0 = get_group_id(0) * 128, m0 = get_group_id(1) * 128;
    const int b = get_group_id(2);
    const long bb0 = b / nb1, bb1 = b % nb1;
    float4 c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    float4 c20 = 0.0f, c21 = 0.0f, c30 = 0.0f, c31 = 0.0f;
    float4 c40 = 0.0f, c41 = 0.0f, c50 = 0.0f, c51 = 0.0f;
    float4 c60 = 0.0f, c61 = 0.0f, c70 = 0.0f, c71 = 0.0f;

    A += aoff + bb0 * sa0 + bb1 * sa1;
    B += boff + bb0 * sb0 + bb1 * sb1;
    C += coff + bb0 * sc0 + bb1 * sc1;
    if (R)
        R += roff + bb0 * sc0 + bb1 * sc1;

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int l = 0; l < 2; l++) {
            const int idx = tid + l * 256;
            const int r = idx >> 2, kq = (idx & 3) * 4;
            const int gm = m0 + r, gk = k0 + kq;
            float4 v = 0.0f;

            if (gm < M && gk < K)
                v = vload4(0, A + (long)gm * lda + gk);
            As[kq][r] = v.x;
            As[kq + 1][r] = v.y;
            As[kq + 2][r] = v.z;
            As[kq + 3][r] = v.w;

            if (transb) {
                const int gn = n0 + r;

                v = 0.0f;
                if (gn < N && gk < K)
                    v = vload4(0, B + (long)gn * ldb + gk);
                Bs[kq][r] = v.x;
                Bs[kq + 1][r] = v.y;
                Bs[kq + 2][r] = v.z;
                Bs[kq + 3][r] = v.w;
            }
            else {
                const int kk = idx >> 5, nq = (idx & 31) * 4;
                const int gk2 = k0 + kk, gn = n0 + nq;

                v = 0.0f;
                if (gk2 < K) {
                    __global const float *bp = B + (long)gk2 * ldb + gn;

                    if (gn + 3 < N)
                        v = vload4(0, bp);
                    else {
                        if (gn < N)
                            v.x = bp[0];
                        if (gn + 1 < N)
                            v.y = bp[1];
                        if (gn + 2 < N)
                            v.z = bp[2];
                    }
                }
                vstore4(v, 0, &Bs[kk][nq]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < 16; k++) {
            const float4 a0 = vload4(0, &As[k][ty * 8]);
            const float4 a1 = vload4(0, &As[k][ty * 8 + 4]);
            const float4 b0 = vload4(0, &Bs[k][tx * 8]);
            const float4 b1 = vload4(0, &Bs[k][tx * 8 + 4]);

            ROW_FMA(0, a0.x);
            ROW_FMA(1, a0.y);
            ROW_FMA(2, a0.z);
            ROW_FMA(3, a0.w);
            ROW_FMA(4, a1.x);
            ROW_FMA(5, a1.y);
            ROW_FMA(6, a1.z);
            ROW_FMA(7, a1.w);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_row8(C, R, bias, m0 + ty * 8 + 0, n0 + tx * 8, M, N, ldc, alpha, act,
               c00, c01);
    store_row8(C, R, bias, m0 + ty * 8 + 1, n0 + tx * 8, M, N, ldc, alpha, act,
               c10, c11);
    store_row8(C, R, bias, m0 + ty * 8 + 2, n0 + tx * 8, M, N, ldc, alpha, act,
               c20, c21);
    store_row8(C, R, bias, m0 + ty * 8 + 3, n0 + tx * 8, M, N, ldc, alpha, act,
               c30, c31);
    store_row8(C, R, bias, m0 + ty * 8 + 4, n0 + tx * 8, M, N, ldc, alpha, act,
               c40, c41);
    store_row8(C, R, bias, m0 + ty * 8 + 5, n0 + tx * 8, M, N, ldc, alpha, act,
               c50, c51);
    store_row8(C, R, bias, m0 + ty * 8 + 6, n0 + tx * 8, M, N, ldc, alpha, act,
               c60, c61);
    store_row8(C, R, bias, m0 + ty * 8 + 7, n0 + tx * 8, M, N, ldc, alpha, act,
               c70, c71);
}

/* Sum and max of val over a work-group of n items (power of two). */
float wg_sum(float val, __local float *scratch, const int n)
{
    const int lid = get_local_id(0);

    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return val;
}

float wg_max(float val, __local float *scratch, const int n)
{
    const int lid = get_local_id(0);

    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return val;
}

/* Row-wise LayerNorm over C channels, Y may alias X. The work-group
 * size (256 for the encoder's wide rows, 64 for the decoder's 64- and
 * 256-channel rows) is the local size chosen by the host. */
__kernel void layernorm(__global const float *X, __global float *Y,
                        __global const float *g, __global const float *bt,
                        const int C, const float eps, const int act)
{
    __local float scratch[256];
    const int lid = get_local_id(0), n = get_local_size(0);
    const long row = get_group_id(0);
    __global const float *x = X + row * C;
    __global float *y = Y + row * C;
    float s = 0.0f, mean, var;
    int c;

    for (c = lid; c < C; c += n)
        s += x[c];
    mean = wg_sum(s, scratch, n) / C;
    s = 0.0f;
    for (c = lid; c < C; c += n) {
        const float d = x[c] - mean;

        s += d * d;
    }
    var = wg_sum(s, scratch, n) / C;
    s = rsqrt(var + eps);
    for (c = lid; c < C; c += n)
        y[c] = apply_act((x[c] - mean) * s * g[c] + bt[c], act);
}

float dot16(const float16 a, const float16 b)
{
    return dot(a.lo.lo, b.lo.lo) + dot(a.lo.hi, b.lo.hi) +
           dot(a.hi.lo, b.hi.lo) + dot(a.hi.hi, b.hi.hi);
}

/* Mask decoder attention with few keys (the 7 prompt tokens), fused
 * scores, softmax and weighting: one work-item per (prompt, head,
 * query). Q/K/V rows hold dint channels with head h at column h * c,
 * c = 16 or 32; per-prompt strides sq/sk/sv may be 0 for tensors shared
 * by all prompts. Online softmax keeps everything in registers. */
__kernel void attn_fewkeys(__global const float *Q, const int sq,
                           __global const float *K, const int sk,
                           __global const float *V, const int sv,
                           __global float *O, const int Nq, const int Nk,
                           const int dint, const int c, const float scale,
                           const int total)
{
    const int gid = get_global_id(0);
    const int i = gid % Nq, ph = gid / Nq;
    const int h = ph % DEC_HEADS, p = ph / DEC_HEADS;
    __global const float *q, *kb, *vb;
    float16 q0, q1 = 0.0f, o0 = 0.0f, o1 = 0.0f;
    float m = -INFINITY, sum = 0.0f;

    if (gid >= total)
        return;
    q = Q + p * sq + i * dint + h * c;
    kb = K + p * sk + h * c;
    vb = V + p * sv + h * c;
    q0 = vload16(0, q);
    if (c > 16)
        q1 = vload16(1, q);

    for (int j = 0; j < Nk; j++) {
        float s = dot16(q0, vload16(0, kb + j * dint)), mn, corr, e;

        if (c > 16)
            s += dot16(q1, vload16(1, kb + j * dint));
        s *= scale;
        mn = fmax(m, s);
        corr = exp(m - mn);
        e = exp(s - mn);
        sum = sum * corr + e;
        o0 = o0 * corr + e * vload16(0, vb + j * dint);
        if (c > 16)
            o1 = o1 * corr + e * vload16(1, vb + j * dint);
        m = mn;
    }
    O += (p * Nq + i) * dint + h * c;
    vstore16(o0 / sum, 0, O);
    if (c > 16)
        vstore16(o1 / sum, 1, O);
}

/* Mask decoder attention of the prompt tokens over the image tokens
 * (head width 16): one 256-item work-group per (prompt, head, query)
 * row. Each item runs an online softmax over its share of the keys;
 * the partial maxima, sums and weighted values are then merged across
 * the work-group, so any number of keys fits. Same arguments as
 * attn_fewkeys; requires c == 16. */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
attn_manykeys(__global const float *Q, const int sq, __global const float *K,
              const int sk, __global const float *V, const int sv,
              __global float *O, const int Nq, const int Nk, const int dint,
              const int c, const float scale, const int total)
{
    __local float part[16][257];
    __local float scratch[256];
    const int lid = get_local_id(0), row = get_group_id(0);
    const int i = row % Nq, ph = row / Nq;
    const int h = ph % DEC_HEADS, p = ph / DEC_HEADS;
    __global const float *kb = K + p * sk + h * c;
    __global const float *vb = V + p * sv + h * c;
    const float16 q = vload16(0, Q + p * sq + i * dint + h * c);
    float16 o = 0.0f;
    float m = -INFINITY, sum = 0.0f, mall, f, inv;
    int j;

    for (j = lid; j < Nk; j += 256) {
        const float s = dot16(q, vload16(0, kb + j * dint)) * scale;
        const float mn = fmax(m, s), corr = exp(m - mn), e = exp(s - mn);

        sum = sum * corr + e;
        o = o * corr + e * vload16(0, vb + j * dint);
        m = mn;
    }
    mall = wg_max(m, scratch, 256);
    f = m == -INFINITY ? 0.0f : exp(m - mall);
    o *= f;
    inv = 1.0f / wg_sum(sum * f, scratch, 256);

    part[0][lid] = o.s0;
    part[1][lid] = o.s1;
    part[2][lid] = o.s2;
    part[3][lid] = o.s3;
    part[4][lid] = o.s4;
    part[5][lid] = o.s5;
    part[6][lid] = o.s6;
    part[7][lid] = o.s7;
    part[8][lid] = o.s8;
    part[9][lid] = o.s9;
    part[10][lid] = o.sa;
    part[11][lid] = o.sb;
    part[12][lid] = o.sc;
    part[13][lid] = o.sd;
    part[14][lid] = o.se;
    part[15][lid] = o.sf;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s)
            for (j = 0; j < 16; j++)
                part[j][lid] += part[j][lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid < 16)
        O[(p * Nq + i) * dint + h * c + lid] = part[lid][0] * inv;
}

/* In-place softmax of attention score rows of L entries, one work-group
 * (64 or 256 items, chosen by the host) per row. */
__kernel void softmax_rows(__global float *S, const int L)
{
    __local float scratch[256];
    const int lid = get_local_id(0), n = get_local_size(0);
    __global float *s = S + (long)get_group_id(0) * L;
    float m = -INFINITY, sum = 0.0f, v;
    int j;

    for (j = lid; j < L; j += n)
        m = fmax(m, s[j]);
    m = wg_max(m, scratch, n);
    for (j = lid; j < L; j += n) {
        v = exp(s[j] - m);
        s[j] = v;
        sum += v;
    }
    v = 1.0f / wg_sum(sum, scratch, n);
    for (j = lid; j < L; j += n)
        s[j] *= v;
}

/* ps x ps / stride-ps patch extraction of a 3 x side x side CHW image
 * into grid^2 x (3 * ps * ps) rows, column order c * ps^2 + ky * ps + kx
 * to match the Conv2d weight layout [E, 3, ps, ps]. */
__kernel void patch_im2col(__global const float *img, __global float *out,
                           const int side, const int grid, const int ps)
{
    const int gid = get_global_id(0), k = 3 * ps * ps;
    const int t = gid / k, col = gid % k;
    const int c = col / (ps * ps), ky = (col % (ps * ps)) / ps, kx = col % ps;
    const int ty = t / grid, tx = t % grid;

    if (t >= grid * grid)
        return;
    out[gid] = img[(c * side + ty * ps + ky) * side + tx * ps + kx];
}

/* 2D rotary position encoding, in place on the queries and keys of
 * fused qkv rows (3 * E floats, head h at column h * hd). Rows come in
 * groups of npos positions (a window, or the whole grid); cs holds the
 * (cos, sin) of every position and frequency, hd / 2 per position. Each
 * channel pair (2j, 2j + 1) is rotated as a complex number. */
__kernel void rope_apply(__global float *qkv, const int E, const int hd,
                         const int npos, __global const float2 *cs,
                         const int total)
{
    const int gid = get_global_id(0), nf = hd / 2;
    const int pr = gid % (E / 2), rest = gid / (E / 2);
    const int qk = rest % 2, r = rest / 2;
    const int h = pr / nf, j = pr % nf;
    __global float *x = qkv + (long)r * 3 * E + qk * E + h * hd + 2 * j;
    float2 rot;
    float a, b;

    if (gid >= total)
        return;
    rot = cs[(r % npos) * nf + j];
    a = x[0];
    b = x[1];
    x[0] = a * rot.x - b * rot.y;
    x[1] = a * rot.y + b * rot.x;
}

/* out[i] = a[i] + b[i % bmod]; out may alias a. */
__kernel void add_bcast(__global float *out, __global const float *a,
                        __global const float *b, const int n, const int bmod)
{
    const int i = get_global_id(0);

    if (i < n)
        out[i] = a[i] + b[i % bmod];
}

/* Partition an H x W token grid into ws x ws windows (zero padded to a
 * multiple of ws), windows ordered row-major. */
__kernel void window_partition(__global const float *X, __global float *out,
                               const int H, const int W, const int C,
                               const int ws, const int nwx, const int total)
{
    const int gid = get_global_id(0);
    const int r = gid / C, c = gid % C;
    const int w = r / (ws * ws), i = r % (ws * ws);
    const int y = (w / nwx) * ws + i / ws, x = (w % nwx) * ws + i % ws;

    if (gid >= total)
        return;
    out[gid] = (y < H && x < W) ? X[(y * W + x) * C + c] : 0.0f;
}

/* Inverse of window_partition dropping the padding, accumulated into X
 * (residual add). */
__kernel void window_unpartition_add(__global float *X,
                                     __global const float *Wd, const int H,
                                     const int W, const int C, const int ws,
                                     const int nwx)
{
    const int gid = get_global_id(0);
    const int t = gid / C, c = gid % C;
    const int y = t / W, x = t % W;
    const int w = (y / ws) * nwx + x / ws, i = (y % ws) * ws + x % ws;

    if (t >= H * W)
        return;
    X[gid] += Wd[(w * ws * ws + i) * C + c];
}

/* 3x3/pad-1 im2col of the tokens t0 .. t0 + total / (9 C) - 1 of a
 * token-major H x W x C map, column order c * 9 + ky * 3 + kx to match
 * the Conv2d weight layout; large maps are processed in bands. */
__kernel void im2col3x3(__global const float *X, __global float *out,
                        const int H, const int W, const int C, const int t0,
                        const int total)
{
    const int gid = get_global_id(0);
    const int t = t0 + gid / (C * 9), col = gid % (C * 9);
    const int c = col / 9, ky = (col % 9) / 3, kx = col % 3;
    const int y = t / W + ky - 1, x = t % W + kx - 1;

    if (gid >= total)
        return;
    out[gid] =
        (y >= 0 && y < H && x >= 0 && x < W) ? X[(y * W + x) * C + c] : 0.0f;
}

/* Scatter the GEMM form of a 2x2/stride-2 ConvTranspose2d into the
 * upsampled map: G is (P * H * W) x (Co * 4) with column
 * co * 4 + dy * 2 + dx, out is P x 2H x 2W x Co. The high-resolution
 * feature map add (2H x 2W x Co, shared by all prompts), when not NULL,
 * is summed in before the activation. */
__kernel void convt_scatter(__global const float *G, __global float *out,
                            __global const float *bias,
                            __global const float *add, const int H,
                            const int W, const int Co, const int total,
                            const int act)
{
    const int gid = get_global_id(0);
    const int co = gid % Co, pix = gid / Co;
    const int ox = pix % (2 * W), rest = pix / (2 * W);
    const int oy = rest % (2 * H), p = rest / (2 * H);
    const int y = oy >> 1, x = ox >> 1, dy = oy & 1, dx = ox & 1;
    float v;

    if (gid >= total)
        return;
    v = G[((p * H + y) * W + x) * (Co * 4) + co * 4 + dy * 2 + dx] + bias[co];
    if (add)
        v += add[gid % (4 * H * W * Co)];
    out[gid] = apply_act(v, act);
}

/* Low-resolution mask logits: out[p, m, pix] = hyper[p, m, :] .
 * U[p, pix, :] for the 3 multimask outputs, 32 channels. */
__kernel void mask_logits(__global const float *U, __global const float *hyper,
                          __global float *out, const int npix, const int total)
{
    const int gid = get_global_id(0);
    const int p = gid / npix, pix = gid % npix;
    __global const float *h = hyper + p * 96;
    float16 u0, u1;

    if (gid >= total)
        return;
    u0 = vload16(0, U + (long)gid * 32);
    u1 = vload16(1, U + (long)gid * 32);
    out[(p * 3 + 0) * npix + pix] =
        dot16(u0, vload16(0, h)) + dot16(u1, vload16(1, h));
    out[(p * 3 + 1) * npix + pix] =
        dot16(u0, vload16(2, h)) + dot16(u1, vload16(3, h));
    out[(p * 3 + 2) * npix + pix] =
        dot16(u0, vload16(4, h)) + dot16(u1, vload16(5, h));
}

/* PyTorch bilinear (align_corners=False) source coordinate. */
float src_coord(float scale, int dst)
{
    const float s = scale * ((float)dst + 0.5f) - 0.5f;

    return s < 0.0f ? 0.0f : s;
}

/* SAM2Transforms.postprocess_masks for selected masks: the L x L logits
 * bilinearly resized straight to the crop's size (SAM 3 squashes the
 * crop to the encoder input, so there is no padding to remove).
 * Writes the binary mask (logit > 0) and accumulates per mask the
 * stability-score counts (logit > +offset, logit > -offset) and the
 * bounding box of the binary mask:
 * stats[6 * k + {0..5}] = inter, union, xmin, ymin, xmax, ymax. */
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void
mask_post(__global const float *lowres, __global const int *sel,
          const int L, const int out_h, const int out_w, const float offset,
          __global uchar *mask, __global int *stats)
{
    __local int l_inter, l_union, l_x0, l_y0, l_x1, l_y1;
    const int ox = get_global_id(0), oy = get_global_id(1);
    const int k = get_global_id(2);
    const int lid = get_local_id(1) * 16 + get_local_id(0);
    __global const float *lr = lowres + (long)sel[k] * L * L;

    if (lid == 0) {
        l_inter = 0;
        l_union = 0;
        l_x0 = 0x7fffffff;
        l_y0 = 0x7fffffff;
        l_x1 = -1;
        l_y1 = -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ox < out_w && oy < out_h) {
        const float sy = src_coord((float)L / out_h, oy);
        const float sx = src_coord((float)L / out_w, ox);
        const int y0 = (int)sy, x0 = (int)sx;
        const int y1 = y0 + (y0 < L - 1), x1 = x0 + (x0 < L - 1);
        const float ly = sy - y0, lx = sx - x0;
        const float v =
            (1.0f - ly) * ((1.0f - lx) * lr[y0 * L + x0] + lx * lr[y0 * L + x1]) +
            ly * ((1.0f - lx) * lr[y1 * L + x0] + lx * lr[y1 * L + x1]);

        mask[((long)k * out_h + oy) * out_w + ox] = v > 0.0f;
        if (v > offset)
            atomic_inc(&l_inter);
        if (v > -offset)
            atomic_inc(&l_union);
        if (v > 0.0f) {
            atomic_min(&l_x0, ox);
            atomic_min(&l_y0, oy);
            atomic_max(&l_x1, ox);
            atomic_max(&l_y1, oy);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        __global int *st = stats + 6 * k;

        if (l_inter)
            atomic_add(&st[0], l_inter);
        if (l_union)
            atomic_add(&st[1], l_union);
        if (l_x1 >= 0) {
            atomic_min(&st[2], l_x0);
            atomic_min(&st[3], l_y0);
            atomic_max(&st[4], l_x1);
            atomic_max(&st[5], l_y1);
        }
    }
}

/* In-place softmax of attention score rows (row = head * Nq + query,
 * heads from h0), one work-group (64 or 256 items) per row of L keys,
 * with optional terms added first: a causal mask (key j > query i is
 * excluded), and SAM 3's box relative position bias, dy[(i * (L / W) +
 * y) * nh + h] + dx[(i * W + x) * nh + h] for key j = y * W + x. */
__kernel void softmax_ext(__global float *S, const int L, const int Nq,
                          const int h0, const int causal,
                          __global const float *dy, __global const float *dx,
                          const int W, const int nh)
{
    __local float scratch[256];
    const int lid = get_local_id(0), n = get_local_size(0);
    const int row = get_group_id(0), i = row % Nq, h = h0 + row / Nq;
    const int H = W > 0 ? L / W : 0;
    __global float *s = S + (long)row * L;
    float m = -INFINITY, sum = 0.0f, v;
    int j;

    for (j = lid; j < L; j += n) {
        v = s[j];
        if (causal && j > i)
            v = -INFINITY;
        else if (dy)
            v += dy[(i * H + j / W) * nh + h] + dx[(i * W + j % W) * nh + h];
        s[j] = v;
        m = fmax(m, v);
    }
    m = wg_max(m, scratch, n);
    for (j = lid; j < L; j += n) {
        v = exp(s[j] - m);
        s[j] = v;
        sum += v;
    }
    v = 1.0f / wg_sum(sum, scratch, n);
    for (j = lid; j < L; j += n)
        s[j] *= v;
}

/* GroupNorm (G groups of C / G channels, statistics over all HW
 * positions of a group) followed by ReLU, in place on a token-major
 * HW x C map: one 256-item work-group per group. */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
groupnorm_relu(__global float *X, const int HW, const int C,
               __global const float *g, __global const float *b,
               const float eps)
{
    __local float scratch[256];
    const int lid = get_local_id(0), grp = get_group_id(0);
    const int cg = C / get_num_groups(0), c0 = grp * cg, total = HW * cg;
    float s = 0.0f, mean, inv;
    int i;

    for (i = lid; i < total; i += 256)
        s += X[(i / cg) * C + c0 + i % cg];
    mean = wg_sum(s, scratch, 256) / total;
    s = 0.0f;
    for (i = lid; i < total; i += 256) {
        const float d = X[(i / cg) * C + c0 + i % cg] - mean;

        s += d * d;
    }
    inv = rsqrt(wg_sum(s, scratch, 256) / total + eps);
    for (i = lid; i < total; i += 256) {
        const int c = c0 + i % cg;
        __global float *x = X + (i / cg) * C + c;

        *x = fmax((*x - mean) * inv * g[c] + b[c], 0.0f);
    }
}

/* out (H x W x C) = A + nearest 2x upsampling of Lo (H/2 x W/2 x C). */
__kernel void up2_add(__global float *out, __global const float *A,
                      __global const float *Lo, const int W, const int C,
                      const int total)
{
    const int gid = get_global_id(0);
    const int c = gid % C, t = gid / C;
    const int y = t / W, x = t % W;

    if (gid >= total)
        return;
    out[gid] = A[gid] + Lo[((y / 2) * (W / 2) + x / 2) * C + c];
}
