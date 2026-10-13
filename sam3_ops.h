/* SPDX-License-Identifier: Unlicense */

/* Device operations shared by the point-prompt model (sam3_model.c) and
 * the text-prompt detector (sam3_text.c). Internal to the module. */

#ifndef I_SAM3_OPENCL_SAM3_OPS_H
#define I_SAM3_OPENCL_SAM3_OPS_H

#include "sam3_model.h"

#define ACT_NONE 0
#define ACT_GELU 1
#define ACT_RELU 2

#define SETARG(k, i, v) ocl_check(clSetKernelArg(k, i, sizeof(v), &(v)), #k)

/* One GEMM operand: buffer, element offset, leading dimension and the
 * two batch strides. */
struct opnd {
    cl_mem buf;
    cl_long off;
    cl_int ld;
    cl_long s0, s1;
};

cl_mem load_dev(struct sam_model *m, const struct pth_file *pf,
                const char *name, int ndim, const long *shape);
cl_mem load_dev_fmt(struct sam_model *m, const struct pth_file *pf, int ndim,
                    const long *shape, const char *fmt, ...);
float *load_host(const struct pth_file *pf, const char *name, int ndim,
                 const long *shape);
void load_host_into(const struct pth_file *pf, const char *name, int ndim,
                    const long *shape, float *dst);
void load_neck(struct sam_model *m, const struct pth_file *pf,
               const char *prefix, struct neck_level lv[3]);
void release_neck(struct neck_level lv[3]);

void run(struct sam_model *m, cl_kernel k, int dim, const size_t *g,
         const size_t *l, const char *what);
void run1d(struct sam_model *m, cl_kernel k, size_t n, const char *what);

struct opnd mat(cl_mem buf, long off, int ld);
struct opnd bmat(cl_mem buf, long off, int ld, long s0, long s1);

/* Batched C = act(alpha * A * op(B) + bias) + R (see the gemm kernel). */
void gemm(struct sam_model *m, int M, int N, int K, struct opnd A,
          struct opnd B, int transb, struct opnd C, int nb, int nb1,
          cl_mem bias, cl_mem R, long roff, float alpha, int act);

/* Y[rows, out] = act(X[rows, in] W^T + b) (+ Y when residual). */
void linear(struct sam_model *m, int rows, int in, int out, cl_mem X, long xoff,
            cl_mem W, cl_mem b, cl_mem Y, long yoff, int act, int residual);

void layernorm(struct sam_model *m, long rows, int C, cl_mem X, cl_mem Y,
               cl_mem g, cl_mem b, float eps, int act);

/* out[i] = a[i] + b[i % bmod]; out may alias a. */
void add_bcast(struct sam_model *m, cl_mem out, cl_mem a, cl_mem b, long n,
               long bmod);

/* 3x3/pad-1 convolution of a side x side x SAM_DIM map. */
void conv3x3(struct sam_model *m, cl_mem X, int side, cl_mem w, cl_mem b,
             cl_mem Y);

/* One level of a SimpleFPN neck on the ViT output in m->x. */
void neck_level(struct sam_model *m, const struct neck_level *n, cl_mem out);

/* y = W x + b for a PyTorch [nout, nin] weight, optional ReLU. */
void host_linear(const float *W, const float *b, const float *x, int nin,
                 int nout, int relu, float *y);

#endif /* I_SAM3_OPENCL_SAM3_OPS_H */
