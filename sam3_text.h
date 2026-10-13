/* SPDX-License-Identifier: Unlicense */

#ifndef I_SAM3_OPENCL_SAM3_TEXT_H
#define I_SAM3_OPENCL_SAM3_TEXT_H

#include "pth_loader.h"
#include "sam3_model.h"

#define TEXT_QUERIES 200 /* Detector object queries. */

struct sam3_text;

/* Load SAM 3's concept detector (detector neck, geometry encoder,
 * fusion encoder, decoder, scoring and segmentation head) and encode
 * the n text prompts with the text encoder, whose weights are freed
 * again afterwards. m must have been loaded without the point head. */
struct sam3_text *text_load(struct sam_model *m, const struct pth_file *pf,
                            int n, char **prompts);

/* Prepare the prompt-independent detector inputs of the image whose
 * backbone output sam_encode_trunk() left in m->x. */
void text_set_image(struct sam3_text *t);

/* Detect prompt k in the current image: writes the queries whose score
 * (object probability times presence probability) exceeds threshold to
 * qidx[] with their scores, best first, and returns their number. */
int text_detect(struct sam3_text *t, int k, float threshold, int *qidx,
                float *score);

/* Mask logits of n <= text_mask_capacity() detected queries of the last
 * text_detect() call, as low-resolution planes 0 .. n - 1 for
 * sam_postprocess(). */
void text_masks(struct sam3_text *t, int n, const int *qidx);
int text_mask_capacity(const struct sam3_text *t);

/* Encoder outputs for debug dumps, token-major: the 72 x 72 x 256
 * detector features and the first prompt's 256-channel text features
 * (returning their token count, padding excluded). */
void text_read_features(struct sam3_text *t, float *img, float *text,
                        int *ntok);

void text_free(struct sam3_text *t);

#endif /* I_SAM3_OPENCL_SAM3_TEXT_H */
