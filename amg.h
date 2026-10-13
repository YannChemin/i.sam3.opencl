/* SPDX-License-Identifier: Unlicense */

#ifndef I_SAM3_OPENCL_AMG_H
#define I_SAM3_OPENCL_AMG_H

#include "image.h"
#include "sam3_model.h"

/* One accepted mask, stored as a byte map over its bounding box in
 * region cell coordinates (inclusive box). */
struct sam_mask {
    int x0, y0, x1, y1;
    unsigned char *m;
    float iou, stability;
    double crop_score; /* 1 / crop area, for cross-crop NMS. */
    long area;
};

struct mask_list {
    int n, cap;
    struct sam_mask *v;
};

/* SAM2AutomaticMaskGenerator parameters (same names and defaults as the
 * Python class), plus the tiling of regions larger than one crop. */
struct amg_params {
    int points_per_side;
    double pred_iou_thresh;
    double stability_thresh;
    double stability_offset;
    double box_nms_thresh;
    int crop_layers;
    double crop_nms_thresh;
    double crop_overlap_ratio;
    int crop_points_downscale;
    int min_region_area;
    int tile_size;
    int tile_overlap;
    const char *dump_dir; /* Debug dumps of encoder input/embedding. */
};

void amg_generate(struct sam_model *m, const struct sam_image *img,
                  const struct amg_params *p, struct mask_list *out);

void mask_list_free(struct mask_list *l);

#endif /* I_SAM3_OPENCL_AMG_H */
