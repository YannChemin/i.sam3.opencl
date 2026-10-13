/* SPDX-License-Identifier: Unlicense */

/* Whole-run progress: one G_percent() bar from the first band read to the
 * last vector written, which MetaMonitor's GRASS hook picks up
 * (https://github.com/YannChemin/MetaMonitor). */

#ifndef I_SAM3_OPENCL_PROGRESS_H
#define I_SAM3_OPENCL_PROGRESS_H

#include <stddef.h>

enum progress_kind {
    PG_READ,   /* Input bands, in millions of cells. */
    PG_LOAD,   /* Checkpoint tensors, in GiB. */
    PG_BUILD,  /* OpenCL program build. */
    PG_ENCODE, /* Image encoder, per crop. */
    PG_DECODE, /* Mask decoder, per batch of point prompts. */
    PG_TEXT,   /* Concept detector and masks, per prompt and crop. */
    PG_CLEAN,  /* Cross-crop NMS and small-region removal. */
    PG_VECTOR, /* Mask painting and vectorization, per run. */
    PG_NKINDS
};

/* Add units of work of one kind to the run's plan. */
void progress_plan(enum progress_kind k, double units);

/* Start units of work of one kind (planned or not), advance through them
 * with progress_part(), and close them with progress_end(). */
void progress_begin(enum progress_kind k, double units);
void progress_part(double done, double total);
void progress_end(void);

/* Bytes read from the checkpoint, advancing an open PG_LOAD stage. */
void progress_bytes(size_t n);

/* Report 100 %. */
void progress_finish(void);

#endif /* I_SAM3_OPENCL_PROGRESS_H */
