/* SPDX-License-Identifier: Unlicense */

#ifndef I_SAM3_OPENCL_OCL_BACKEND_H
#define I_SAM3_OPENCL_OCL_BACKEND_H

/* Target the OpenCL 1.2 API, the common denominator of PoCL, Mesa
 * rusticl/Clover and the discrete GPU vendor ICDs. */
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>

#include <stddef.h>

/* Device profiles (profile= option). They choose the buffer strategy
 * and the GEMM kernels:
 * - cpu:  host-memory device (PoCL): mapped buffers, GEMM kernels
 *         without local memory, vectorised along the contiguous axis.
 * - igpu: GPU sharing host memory: zero-copy buffers (allocated in host
 *         memory, written and read by mapping instead of copying).
 * - dgpu: GPU with its own memory on internal PCIe: device buffers.
 * - egpu: GPU behind a slow link (Thunderbolt, PoCL-Remote): device
 *         buffers, transfers through a pinned staging buffer. */
enum ocl_arch { OCL_ARCH_CPU, OCL_ARCH_IGPU, OCL_ARCH_DGPU, OCL_ARCH_EGPU };

struct ocl_backend {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_program program_h; /* GEMMs with half-precision B (precision=fp16). */
    char device_name[256];
    char platform_name[256];
    cl_ulong max_alloc;
    cl_ulong global_mem;
    int is_gpu;
    int unified; /* CL_DEVICE_HOST_UNIFIED_MEMORY. */
    int arch;    /* enum ocl_arch. */
    int fp16w;   /* Weight matrices stored as half. */
    int profile; /* Per-kernel timing (I_SAM3_OPENCL_PROFILE set). */
    cl_mem staging; /* Pinned staging buffer (egpu). */
    void *staging_ptr;
    size_t staging_size;
};

/* Select a device and build the embedded kernels.
 *
 * device_opt is "auto" (first GPU, else first CPU device), "gpu" or
 * "cpu". platform_opt, when not NULL, restricts the search to platforms
 * whose name contains that substring (case-insensitive), e.g. "rusticl"
 * or "Portable". profile_opt is "auto" or one of cpu, igpu, dgpu, egpu
 * (see enum ocl_arch); precision_opt is "fp32" or "fp16" (weight
 * matrices stored as half, arithmetic in float). Fails with
 * G_fatal_error() if no device matches: this module has no non-OpenCL
 * fallback. */
void ocl_init(struct ocl_backend *ocl, const char *device_opt,
              const char *platform_opt, const char *profile_opt,
              const char *precision_opt);

const char *ocl_arch_name(int arch);

/* Upload a weight matrix of count floats used only as GEMM operand B:
 * stored as half with precision=fp16, else as float. */
cl_mem ocl_upload_weight(struct ocl_backend *ocl, const float *src,
                         size_t count, const char *what);

/* Non-zero when m holds half-precision weights. */
int ocl_is_half(cl_mem m);

/* Kernels allowed to receive half-precision buffers. */
void ocl_allow_half(cl_kernel k);

/* clSetKernelArg, refusing a half-precision buffer for a kernel that
 * reads float (an internal error, reported with the kernel name). */
void ocl_setarg(cl_kernel k, cl_uint i, size_t size, const void *v,
                const char *what);

void ocl_free(struct ocl_backend *ocl);

/* Human-readable name of an OpenCL error code. */
const char *ocl_errstr(cl_int err);

/* Fatal error helper: aborts with a message naming the failed call. */
void ocl_check(cl_int err, const char *what);

/* Allocate a device buffer of n bytes, fatal on failure. */
cl_mem ocl_alloc(struct ocl_backend *ocl, size_t n, const char *what);

/* Allocate and fill a device buffer from host memory. */
cl_mem ocl_upload(struct ocl_backend *ocl, const void *src, size_t n,
                  const char *what);

void ocl_write(struct ocl_backend *ocl, cl_mem dst, size_t off, const void *src,
               size_t n);
void ocl_read(struct ocl_backend *ocl, cl_mem src, size_t off, void *dst,
              size_t n);

cl_kernel ocl_kernel(struct ocl_backend *ocl, const char *name);

/* Kernel from the half-precision GEMM program (precision=fp16). */
cl_kernel ocl_kernel_h(struct ocl_backend *ocl, const char *name);

void ocl_release(cl_mem *m);

#endif /* I_SAM3_OPENCL_OCL_BACKEND_H */
