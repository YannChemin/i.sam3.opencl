/* SPDX-License-Identifier: Unlicense */

#ifndef I_SAM3_OPENCL_VECTORIZE_H
#define I_SAM3_OPENCL_VECTORIZE_H

#include "amg.h"
#include "image.h"

/* How overlapping masks share cells. */
enum overlap_rule {
    OVERLAP_SMALL, /* The smallest covering mask wins. */
    OVERLAP_LARGE, /* The largest covering mask wins. */
    OVERLAP_SCORE  /* The highest predicted IoU wins. */
};

struct vect_params {
    enum overlap_rule overlap;
    long minsize;    /* Cells; smaller pieces are merged into neighbours. */
    long maxsize;    /* Cells; larger masks are discarded, 0 = no limit. */
    double simplify; /* Douglas-Peucker tolerance in map units, 0 = off. */
    const char *vector_out;
    const char *raster_out; /* Optional label raster, may be NULL. */
    const char *source;     /* Input description for the history. */
    char **labels;          /* Text prompts by mask label, or NULL. */
};

/* Turn the (overlapping) SAM masks into a clean, non-overlapping vector
 * area map with one attribute row per segment. */
void vectorize_masks(const struct mask_list *masks, const struct sam_image *img,
                     const struct vect_params *vp);

#endif /* I_SAM3_OPENCL_VECTORIZE_H */
