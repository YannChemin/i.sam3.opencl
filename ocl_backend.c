/****************************************************************************
 *
 * MODULE:       i.sam3.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      OpenCL platform/device selection, program build and
 *               buffer helpers.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "ocl_backend.h"

/* sam3_kernels.cl, turned into a string literal by the Makefile. */
static const char *sam3_kernel_source =
#include "sam3_kernels_cl.h"
    ;

const char *ocl_errstr(cl_int err)
{
    switch (err) {
    case CL_SUCCESS:
        return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
    default:
        return "unknown OpenCL error";
    }
}

void ocl_check(cl_int err, const char *what)
{
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: %s failed: %s (%d)"), what, ocl_errstr(err),
                      err);
}

static int contains_nocase(const char *hay, const char *needle)
{
    size_t n = strlen(needle), i, j;

    for (i = 0; hay[i]; i++) {
        for (j = 0; j < n && hay[i + j]; j++)
            if (tolower((unsigned char)hay[i + j]) !=
                tolower((unsigned char)needle[j]))
                break;
        if (j == n)
            return 1;
    }
    return n == 0;
}

struct candidate {
    cl_platform_id platform;
    cl_device_id device;
};

/* Append the devices of the given type found on platforms matching
 * platform_opt, in platform order. */
static int list_devices(cl_device_type type, const char *platform_opt,
                        struct candidate *out, int n, int max)
{
    cl_platform_id platforms[16];
    cl_device_id devices[8];
    cl_uint np, nd, p, d;
    char pname[256];
    int k;

    if (clGetPlatformIDs(16, platforms, &np) != CL_SUCCESS)
        return n;
    for (p = 0; p < np; p++) {
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(pname), pname,
                          NULL);
        if (platform_opt && !contains_nocase(pname, platform_opt))
            continue;
        if (clGetDeviceIDs(platforms[p], type, 8, devices, &nd) != CL_SUCCESS)
            continue;
        for (d = 0; d < nd && n < max; d++) {
            for (k = 0; k < n; k++)
                if (out[k].device == devices[d])
                    break;
            if (k == n) {
                out[n].platform = platforms[p];
                out[n++].device = devices[d];
            }
        }
    }
    return n;
}

/* Set up context, queue and program on one device. Returns 0 and fills
 * log (if not NULL) when the kernels do not build there. */
static int try_device(struct ocl_backend *ocl, const struct candidate *c,
                      char **log)
{
    const char *src = sam3_kernel_source;
    size_t len = strlen(src);
    char version[128];
    const char *opts;
    char opts_h[160];
    cl_device_type type;
    cl_bool unified;
    cl_int err;

    ocl->platform = c->platform;
    ocl->device = c->device;
    clGetPlatformInfo(c->platform, CL_PLATFORM_NAME, sizeof(ocl->platform_name),
                      ocl->platform_name, NULL);
    clGetDeviceInfo(c->device, CL_DEVICE_NAME, sizeof(ocl->device_name),
                    ocl->device_name, NULL);
    clGetDeviceInfo(c->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
                    &ocl->max_alloc, NULL);
    clGetDeviceInfo(c->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                    &ocl->global_mem, NULL);
    clGetDeviceInfo(c->device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    ocl->is_gpu = (type & CL_DEVICE_TYPE_GPU) != 0;
    unified = CL_FALSE;
    clGetDeviceInfo(c->device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified),
                    &unified, NULL);
    ocl->unified = unified == CL_TRUE;

    /* The kernels are OpenCL C 1.1; request 1.2 where available. */
    version[0] = '\0';
    clGetDeviceInfo(c->device, CL_DEVICE_OPENCL_C_VERSION, sizeof(version),
                    version, NULL);
    opts = strncmp(version, "OpenCL C 1.1", 12) == 0 ||
                   strncmp(version, "OpenCL C 1.0", 12) == 0
               ? "-cl-std=CL1.1 -cl-mad-enable"
               : "-cl-std=CL1.2 -cl-mad-enable";

    ocl->context = clCreateContext(NULL, 1, &c->device, NULL, NULL, &err);
    ocl_check(err, "clCreateContext");
    /* Developer aid: I_SAM3_OPENCL_PROFILE=1 times every kernel. */
    ocl->profile = getenv("I_SAM3_OPENCL_PROFILE") != NULL;
    ocl->queue = clCreateCommandQueue(
        ocl->context, c->device, ocl->profile ? CL_QUEUE_PROFILING_ENABLE : 0,
        &err);
    ocl_check(err, "clCreateCommandQueue");
    ocl->program = clCreateProgramWithSource(ocl->context, 1, &src, &len, &err);
    ocl_check(err, "clCreateProgramWithSource");

    err = clBuildProgram(ocl->program, 1, &c->device, opts, NULL, NULL);
    if (err == CL_SUCCESS && ocl->fp16w) {
        /* Second program: only the GEMMs, with operand B in half. */
        snprintf(opts_h, sizeof(opts_h), "%s -DSAM_BHALF -DSAM_GEMM_ONLY",
                 opts);
        ocl->program_h =
            clCreateProgramWithSource(ocl->context, 1, &src, &len, &err);
        ocl_check(err, "clCreateProgramWithSource");
        err = clBuildProgram(ocl->program_h, 1, &c->device, opts_h, NULL,
                             NULL);
        if (err != CL_SUCCESS) {
            if (log) {
                size_t loglen = 0;

                clGetProgramBuildInfo(ocl->program_h, c->device,
                                      CL_PROGRAM_BUILD_LOG, 0, NULL, &loglen);
                *log = G_malloc(loglen + 1);
                clGetProgramBuildInfo(ocl->program_h, c->device,
                                      CL_PROGRAM_BUILD_LOG, loglen, *log,
                                      NULL);
                (*log)[loglen] = '\0';
                log = NULL;
            }
            clReleaseProgram(ocl->program_h);
            ocl->program_h = NULL;
        }
    }
    if (err == CL_SUCCESS)
        return 1;

    if (log) {
        size_t loglen = 0;

        clGetProgramBuildInfo(ocl->program, c->device, CL_PROGRAM_BUILD_LOG, 0,
                              NULL, &loglen);
        *log = G_malloc(loglen + 1);
        clGetProgramBuildInfo(ocl->program, c->device, CL_PROGRAM_BUILD_LOG,
                              loglen, *log, NULL);
        (*log)[loglen] = '\0';
    }
    clReleaseProgram(ocl->program);
    clReleaseCommandQueue(ocl->queue);
    clReleaseContext(ocl->context);
    ocl->program = NULL;
    ocl->queue = NULL;
    ocl->context = NULL;
    return 0;
}

const char *ocl_arch_name(int arch)
{
    static const char *names[] = {"cpu", "igpu", "dgpu", "egpu"};

    return arch >= 0 && arch <= OCL_ARCH_EGPU ? names[arch] : "?";
}

#ifndef CL_DEVICE_PCI_BUS_INFO_KHR
#define CL_DEVICE_PCI_BUS_INFO_KHR 0x410F
#endif
#ifndef CL_DEVICE_PCI_BUS_ID_NV
#define CL_DEVICE_PCI_BUS_ID_NV 0x4008
#define CL_DEVICE_PCI_SLOT_ID_NV 0x4009
#endif

/* Non-zero when Linux marks the device's PCI function as removable
 * (Thunderbolt / USB4 eGPU). The PCI address comes from
 * cl_khr_pci_bus_info, or from the NVIDIA query; 0 when unknown. */
static int pci_removable(const struct ocl_backend *ocl)
{
    struct {
        cl_uint domain, bus, device, function;
    } pci;
    char path[96], buf[32];
    cl_uint nv_bus, nv_slot;
    FILE *f;
    int r = 0;

    if (clGetDeviceInfo(ocl->device, CL_DEVICE_PCI_BUS_INFO_KHR, sizeof(pci),
                        &pci, NULL) != CL_SUCCESS) {
        if (clGetDeviceInfo(ocl->device, CL_DEVICE_PCI_BUS_ID_NV,
                            sizeof(nv_bus), &nv_bus, NULL) != CL_SUCCESS ||
            clGetDeviceInfo(ocl->device, CL_DEVICE_PCI_SLOT_ID_NV,
                            sizeof(nv_slot), &nv_slot, NULL) != CL_SUCCESS)
            return 0;
        pci.domain = 0;
        pci.bus = nv_bus;
        pci.device = nv_slot >> 3;
        pci.function = nv_slot & 7;
    }
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.%x/removable",
             pci.domain, pci.bus, pci.device, pci.function);
    f = fopen(path, "r");
    if (!f)
        return 0;
    if (fgets(buf, sizeof(buf), f))
        r = strncmp(buf, "removable", 9) == 0;
    fclose(f);
    return r;
}

/* Profile of the selected device: forced by profile_opt, else detected. */
static void select_arch(struct ocl_backend *ocl, const char *profile_opt)
{
    const char *why;

    if (profile_opt && strcmp(profile_opt, "auto") != 0) {
        if (strcmp(profile_opt, "cpu") == 0)
            ocl->arch = OCL_ARCH_CPU;
        else if (strcmp(profile_opt, "igpu") == 0)
            ocl->arch = OCL_ARCH_IGPU;
        else if (strcmp(profile_opt, "dgpu") == 0)
            ocl->arch = OCL_ARCH_DGPU;
        else if (strcmp(profile_opt, "egpu") == 0)
            ocl->arch = OCL_ARCH_EGPU;
        else
            G_fatal_error(_("Invalid profile option <%s>"), profile_opt);
        why = _("forced by profile=");
    }
    else if (!ocl->is_gpu) {
        ocl->arch = OCL_ARCH_CPU;
        why = _("CPU device");
    }
    else if (ocl->unified) {
        ocl->arch = OCL_ARCH_IGPU;
        why = _("GPU sharing host memory");
    }
    else if (pci_removable(ocl)) {
        ocl->arch = OCL_ARCH_EGPU;
        why = _("GPU on a removable PCI link (Thunderbolt/USB4)");
    }
    else {
        ocl->arch = OCL_ARCH_DGPU;
        why = _("GPU with its own memory");
    }
    G_message(_("OpenCL profile: %s (%s), weights in %s"),
              ocl_arch_name(ocl->arch), why, ocl->fp16w ? "fp16" : "fp32");
}

void ocl_init(struct ocl_backend *ocl, const char *device_opt,
              const char *platform_opt, const char *profile_opt,
              const char *precision_opt)
{
    struct candidate cand[32];
    char *log = NULL;
    int n = 0, i;

    memset(ocl, 0, sizeof(*ocl));
    if (platform_opt && !*platform_opt)
        platform_opt = NULL;
    if (!precision_opt || strcmp(precision_opt, "fp32") == 0)
        ocl->fp16w = 0;
    else if (strcmp(precision_opt, "fp16") == 0)
        ocl->fp16w = 1;
    else
        G_fatal_error(_("Invalid precision option <%s>"), precision_opt);

    if (!device_opt || strcmp(device_opt, "auto") == 0) {
        n = list_devices(CL_DEVICE_TYPE_GPU, platform_opt, cand, n, 32);
        n = list_devices(CL_DEVICE_TYPE_CPU, platform_opt, cand, n, 32);
        n = list_devices(CL_DEVICE_TYPE_ALL, platform_opt, cand, n, 32);
    }
    else if (strcmp(device_opt, "gpu") == 0)
        n = list_devices(CL_DEVICE_TYPE_GPU, platform_opt, cand, n, 32);
    else if (strcmp(device_opt, "cpu") == 0)
        n = list_devices(CL_DEVICE_TYPE_CPU, platform_opt, cand, n, 32);
    else
        G_fatal_error(_("Invalid device option <%s>"), device_opt);

    if (n == 0)
        G_fatal_error(_("OpenCL: no usable %s device found%s%s"),
                      device_opt ? device_opt : "auto",
                      platform_opt ? _(" on platforms matching ") : "",
                      platform_opt ? platform_opt : "");

    /* The same GPU can be exposed by several platforms (e.g. Mesa Clover
     * and rusticl) of which only some compile the kernels: take the
     * first one that does. */
    for (i = 0; i < n; i++) {
        G_free(log);
        log = NULL;
        if (try_device(ocl, &cand[i], &log))
            break;
        G_warning(_("OpenCL: kernels do not build on <%s> [%s], trying the "
                    "next device"),
                  ocl->device_name, ocl->platform_name);
        G_debug(1, "build log:\n%s", log);
    }
    if (i == n)
        G_fatal_error(_("OpenCL: kernel build failed on every candidate "
                        "device; last build log:\n%s"),
                      log ? log : "");
    G_free(log);

    G_message(_("OpenCL device: %s [%s], %lu MiB, max buffer %lu MiB"),
              ocl->device_name, ocl->platform_name,
              (unsigned long)(ocl->global_mem >> 20),
              (unsigned long)(ocl->max_alloc >> 20));
    select_arch(ocl, profile_opt);
}

void ocl_free(struct ocl_backend *ocl)
{
    if (ocl->staging) {
        clEnqueueUnmapMemObject(ocl->queue, ocl->staging, ocl->staging_ptr, 0,
                                NULL, NULL);
        clFinish(ocl->queue);
        clReleaseMemObject(ocl->staging);
    }
    if (ocl->program_h)
        clReleaseProgram(ocl->program_h);
    if (ocl->program)
        clReleaseProgram(ocl->program);
    if (ocl->queue)
        clReleaseCommandQueue(ocl->queue);
    if (ocl->context)
        clReleaseContext(ocl->context);
    memset(ocl, 0, sizeof(*ocl));
}

cl_mem ocl_alloc(struct ocl_backend *ocl, size_t n, const char *what)
{
    cl_int err;
    cl_mem m;

    if (n == 0)
        n = 4;
    if (n > ocl->max_alloc)
        G_fatal_error(_("OpenCL: buffer <%s> needs %lu MiB, more than the "
                        "device maximum allocation of %lu MiB (reduce "
                        "batch= or tile_size=)"),
                      what, (unsigned long)(n >> 20),
                      (unsigned long)(ocl->max_alloc >> 20));
    /* A CPU device gets buffers in host memory. On an integrated GPU the
     * driver places them: host-allocated buffers made Intel's kernels
     * about 15 % slower; transfers are mapped instead (ocl_write). */
    m = clCreateBuffer(ocl->context,
                       CL_MEM_READ_WRITE | (ocl->arch == OCL_ARCH_CPU
                                                ? CL_MEM_ALLOC_HOST_PTR
                                                : 0),
                       n, NULL, &err);
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: allocating %lu MiB for <%s> failed: %s"),
                      (unsigned long)(n >> 20), what, ocl_errstr(err));
    return m;
}

cl_mem ocl_upload(struct ocl_backend *ocl, const void *src, size_t n,
                  const char *what)
{
    cl_mem m = ocl_alloc(ocl, n, what);

    ocl_write(ocl, m, 0, src, n);
    return m;
}

/* Transfers of at least this size go through the pinned staging buffer
 * on an eGPU. */
#define STAGING_MIN (64 << 10)

/* Pinned host buffer of at least n bytes, kept mapped (egpu). */
static void *staging(struct ocl_backend *ocl, size_t n)
{
    cl_int err;

    if (n <= ocl->staging_size)
        return ocl->staging_ptr;
    if (ocl->staging) {
        clEnqueueUnmapMemObject(ocl->queue, ocl->staging, ocl->staging_ptr, 0,
                                NULL, NULL);
        clFinish(ocl->queue);
        clReleaseMemObject(ocl->staging);
    }
    ocl->staging = clCreateBuffer(ocl->context,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, n,
                                  NULL, &err);
    ocl_check(err, "clCreateBuffer (staging)");
    ocl->staging_ptr =
        clEnqueueMapBuffer(ocl->queue, ocl->staging, CL_TRUE,
                           CL_MAP_READ | CL_MAP_WRITE, 0, n, 0, NULL, NULL, &err);
    ocl_check(err, "clEnqueueMapBuffer (staging)");
    ocl->staging_size = n;
    return ocl->staging_ptr;
}

void ocl_write(struct ocl_backend *ocl, cl_mem dst, size_t off, const void *src,
               size_t n)
{
    cl_int err;
    void *p;

    if (ocl->arch == OCL_ARCH_CPU || ocl->arch == OCL_ARCH_IGPU) {
        p = clEnqueueMapBuffer(ocl->queue, dst, CL_TRUE, CL_MAP_WRITE, off, n,
                               0, NULL, NULL, &err);
        ocl_check(err, "clEnqueueMapBuffer");
        memcpy(p, src, n);
        ocl_check(clEnqueueUnmapMemObject(ocl->queue, dst, p, 0, NULL, NULL),
                  "clEnqueueUnmapMemObject");
        return;
    }
    if (ocl->arch == OCL_ARCH_EGPU && n >= STAGING_MIN) {
        p = staging(ocl, n);
        memcpy(p, src, n);
        src = p;
    }
    ocl_check(clEnqueueWriteBuffer(ocl->queue, dst, CL_TRUE, off, n, src, 0,
                                   NULL, NULL),
              "clEnqueueWriteBuffer");
}

void ocl_read(struct ocl_backend *ocl, cl_mem src, size_t off, void *dst,
              size_t n)
{
    cl_int err;
    void *p;

    if (ocl->arch == OCL_ARCH_CPU || ocl->arch == OCL_ARCH_IGPU) {
        p = clEnqueueMapBuffer(ocl->queue, src, CL_TRUE, CL_MAP_READ, off, n, 0,
                               NULL, NULL, &err);
        ocl_check(err, "clEnqueueMapBuffer");
        memcpy(dst, p, n);
        ocl_check(clEnqueueUnmapMemObject(ocl->queue, src, p, 0, NULL, NULL),
                  "clEnqueueUnmapMemObject");
        return;
    }
    if (ocl->arch == OCL_ARCH_EGPU && n >= STAGING_MIN) {
        p = staging(ocl, n);
        ocl_check(clEnqueueReadBuffer(ocl->queue, src, CL_TRUE, off, n, p, 0,
                                      NULL, NULL),
                  "clEnqueueReadBuffer");
        memcpy(dst, p, n);
        return;
    }
    ocl_check(clEnqueueReadBuffer(ocl->queue, src, CL_TRUE, off, n, dst, 0,
                                  NULL, NULL),
              "clEnqueueReadBuffer");
}

/* Half-precision weights: IEEE binary16, round to nearest even. */
static uint16_t f32_to_f16(float f)
{
    uint32_t x, sign, mant;
    int exp;

    memcpy(&x, &f, 4);
    sign = (x >> 16) & 0x8000;
    exp = (int)((x >> 23) & 0xff) - 127 + 15;
    mant = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) /* Inf or NaN. */
        return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (exp >= 31) /* Overflow to infinity. */
        return (uint16_t)(sign | 0x7c00);
    if (exp <= 0) { /* Subnormal half, or zero. */
        uint32_t m, shift, halfway, rem, r;

        if (exp < -10)
            return (uint16_t)sign;
        m = mant | 0x800000;
        shift = (uint32_t)(14 - exp);
        r = m >> shift;
        rem = m & ((1u << shift) - 1);
        halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (r & 1)))
            r++;
        return (uint16_t)(sign | r);
    }
    {
        uint32_t r = ((uint32_t)exp << 10) | (mant >> 13);
        uint32_t rest = mant & 0x1fff;

        /* Round to nearest, ties to even (a carry into the exponent is
         * the correct result, up to infinity). */
        if (rest > 0x1000 || (rest == 0x1000 && (r & 1)))
            r++;
        return (uint16_t)(sign | r);
    }
}

/* Registries of half-precision buffers and of the kernels allowed to
 * read them (a handful of kernels, a few hundred weight buffers). */
static cl_mem *half_bufs;
static int n_half_bufs, cap_half_bufs;
static cl_kernel half_kernels[16];
static int n_half_kernels;

cl_mem ocl_upload_weight(struct ocl_backend *ocl, const float *src,
                         size_t count, const char *what)
{
    uint16_t *h;
    cl_mem m;
    size_t i;

    if (!ocl->fp16w)
        return ocl_upload(ocl, src, count * sizeof(float), what);
    h = G_malloc(count * sizeof(uint16_t));
    for (i = 0; i < count; i++)
        h[i] = f32_to_f16(src[i]);
    m = ocl_upload(ocl, h, count * sizeof(uint16_t), what);
    G_free(h);
    if (n_half_bufs == cap_half_bufs) {
        cap_half_bufs = cap_half_bufs ? 2 * cap_half_bufs : 256;
        half_bufs = G_realloc(half_bufs, cap_half_bufs * sizeof(cl_mem));
    }
    half_bufs[n_half_bufs++] = m;
    return m;
}

int ocl_is_half(cl_mem m)
{
    int i;

    for (i = 0; m && i < n_half_bufs; i++)
        if (half_bufs[i] == m)
            return 1;
    return 0;
}

void ocl_allow_half(cl_kernel k)
{
    if (n_half_kernels < 16)
        half_kernels[n_half_kernels++] = k;
}

void ocl_setarg(cl_kernel k, cl_uint i, size_t size, const void *v,
                const char *what)
{
    if (n_half_bufs && size == sizeof(cl_mem) && ocl_is_half(*(cl_mem *)v)) {
        int j;

        for (j = 0; j < n_half_kernels && half_kernels[j] != k; j++)
            ;
        if (j == n_half_kernels)
            G_fatal_error(_("OpenCL: internal error: half-precision weights "
                            "passed to a float kernel (%s, argument %u); "
                            "use precision=fp32"),
                          what, i);
    }
    ocl_check(clSetKernelArg(k, i, size, v), what);
}

cl_kernel ocl_kernel(struct ocl_backend *ocl, const char *name)
{
    cl_int err;
    cl_kernel k = clCreateKernel(ocl->program, name, &err);

    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: kernel <%s> not found: %s"), name,
                      ocl_errstr(err));
    return k;
}

cl_kernel ocl_kernel_h(struct ocl_backend *ocl, const char *name)
{
    cl_int err;
    cl_kernel k;

    if (!ocl->program_h)
        G_fatal_error(_("OpenCL: half-precision program not built"));
    k = clCreateKernel(ocl->program_h, name, &err);
    if (err != CL_SUCCESS)
        G_fatal_error(_("OpenCL: kernel <%s> not found: %s"), name,
                      ocl_errstr(err));
    ocl_allow_half(k);
    return k;
}

void ocl_release(cl_mem *m)
{
    int i;

    if (*m) {
        for (i = 0; i < n_half_bufs; i++)
            if (half_bufs[i] == *m) {
                half_bufs[i] = half_bufs[--n_half_bufs];
                break;
            }
        clReleaseMemObject(*m);
        *m = NULL;
    }
}
