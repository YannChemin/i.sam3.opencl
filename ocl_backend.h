/* SPDX-License-Identifier: Unlicense */

#ifndef I_SAM3_OPENCL_OCL_BACKEND_H
#define I_SAM3_OPENCL_OCL_BACKEND_H

/* Target the OpenCL 1.2 API, the common denominator of PoCL, Mesa
 * rusticl/Clover and the discrete GPU vendor ICDs. */
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>

#include <stddef.h>

struct ocl_backend {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    char device_name[256];
    char platform_name[256];
    cl_ulong max_alloc;
    cl_ulong global_mem;
    int is_gpu;
    int profile; /* Per-kernel timing (I_SAM3_OPENCL_PROFILE set). */
};

/* Select a device and build the embedded kernels.
 *
 * device_opt is "auto" (first GPU, else first CPU device), "gpu" or
 * "cpu". platform_opt, when not NULL, restricts the search to platforms
 * whose name contains that substring (case-insensitive), e.g. "rusticl"
 * or "Portable". Fails with G_fatal_error() if no device matches: this
 * module has no non-OpenCL fallback. */
void ocl_init(struct ocl_backend *ocl, const char *device_opt,
              const char *platform_opt);

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

void ocl_release(cl_mem *m);

#endif /* I_SAM3_OPENCL_OCL_BACKEND_H */
