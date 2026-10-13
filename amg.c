/****************************************************************************
 *
 * MODULE:       i.sam3.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Automatic mask generation with SAM 3's point-prompt head:
 *               a C port of sam2's SAM2AutomaticMaskGenerator (point grid
 *               prompts, IoU and stability filtering, box NMS, crop
 *               layers), extended with small region postprocessing and
 *               tiling of large regions.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "amg.h"
#include "progress.h"
#include "tokenizer.h"

/* Masks whose box lies within this many cells of an inner crop edge are
 * discarded (is_box_near_crop_edge atol). */
#define EDGE_ATOL 20.0

static double now(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

struct crop {
    int x0, y0, x1, y1; /* Region cells, x1/y1 exclusive. */
    int layer;
};

static void list_push(struct mask_list *l, const struct sam_mask *mk)
{
    if (l->n == l->cap) {
        l->cap = l->cap ? 2 * l->cap : 256;
        l->v = G_realloc(l->v, l->cap * sizeof(struct sam_mask));
    }
    l->v[l->n++] = *mk;
}

void mask_list_free(struct mask_list *l)
{
    int i;

    for (i = 0; i < l->n; i++)
        G_free(l->v[i].m);
    G_free(l->v);
    memset(l, 0, sizeof(*l));
}

/* Crop boxes. */

static int tile_starts(int len, int tile, int overlap, int *starts)
{
    int n = 0, x, step = tile - overlap;

    if (len <= tile) {
        starts[0] = 0;
        return 1;
    }
    for (x = 0;; x += step) {
        if (x + tile >= len) {
            starts[n++] = len - tile;
            break;
        }
        starts[n++] = x;
    }
    return n;
}

/* sam2.utils.amg.generate_crop_boxes over a w x h tile. */
static void crop_boxes(int w, int h, int n_layers, double ratio, int ox, int oy,
                       struct crop **v, int *n, int *cap)
{
    int short_side = w < h ? w : h, layer, i, j;

#define PUSH_CROP(a, b, c, d, l)                                               \
    do {                                                                       \
        if (*n == *cap) {                                                      \
            *cap = *cap ? 2 * *cap : 16;                                       \
            *v = G_realloc(*v, *cap * sizeof(struct crop));                    \
        }                                                                      \
        (*v)[*n].x0 = ox + (a);                                                \
        (*v)[*n].y0 = oy + (b);                                                \
        (*v)[*n].x1 = ox + (c);                                                \
        (*v)[*n].y1 = oy + (d);                                                \
        (*v)[(*n)++].layer = (l);                                              \
    } while (0)

    PUSH_CROP(0, 0, w, h, 0);
    for (layer = 0; layer < n_layers; layer++) {
        int nc = 1 << (layer + 1);
        int overlap = (int)(ratio * short_side * (2.0 / nc));
        int cw = (int)ceil((double)(overlap * (nc - 1) + w) / nc);
        int ch = (int)ceil((double)(overlap * (nc - 1) + h) / nc);

        for (i = 0; i < nc; i++) {
            int x0 = (int)((cw - overlap) * i);

            for (j = 0; j < nc; j++) {
                int y0 = (int)((ch - overlap) * j);

                PUSH_CROP(x0, y0, x0 + cw < w ? x0 + cw : w,
                          y0 + ch < h ? y0 + ch : h, layer + 1);
            }
        }
    }
#undef PUSH_CROP
}

/* Non-maximum suppression over mask boxes. */

struct order {
    int idx;
    double s1, s2;
};

static int cmp_order(const void *a, const void *b)
{
    const struct order *x = a, *y = b;

    if (x->s1 != y->s1)
        return x->s1 > y->s1 ? -1 : 1;
    if (x->s2 != y->s2)
        return x->s2 > y->s2 ? -1 : 1;
    return x->idx - y->idx;
}

static double box_iou(const struct sam_mask *a, const struct sam_mask *b)
{
    /* torchvision box_iou on the inclusive xyxy boxes SAM produces. */
    double aa = (double)(a->x1 - a->x0) * (a->y1 - a->y0);
    double ab = (double)(b->x1 - b->x0) * (b->y1 - b->y0);
    double iw = fmin(a->x1, b->x1) - fmax(a->x0, b->x0);
    double ih = fmin(a->y1, b->y1) - fmax(a->y0, b->y0);
    double inter = (iw > 0 ? iw : 0) * (ih > 0 ? ih : 0);
    double uni = aa + ab - inter;

    return uni > 0 ? inter / uni : 0.0;
}

/* Keep masks in decreasing (s1, s2) order, dropping any whose box IoU
 * with an already kept one exceeds thresh. Frees dropped masks. */
static void nms(struct mask_list *l, const double *s1, const double *s2,
                double thresh)
{
    struct order *o = G_malloc(l->n * sizeof(struct order));
    char *keep = G_calloc(l->n, 1), *dead = G_calloc(l->n, 1);
    struct sam_mask *nv;
    int i, j, n = 0;

    for (i = 0; i < l->n; i++) {
        o[i].idx = i;
        o[i].s1 = s1[i];
        o[i].s2 = s2 ? s2[i] : 0.0;
    }
    qsort(o, l->n, sizeof(struct order), cmp_order);
    for (i = 0; i < l->n; i++) {
        int a = o[i].idx;

        if (dead[a])
            continue;
        keep[a] = 1;
        for (j = i + 1; j < l->n; j++) {
            int b = o[j].idx;

            /* Masks of different text prompts never suppress each
             * other (batched_nms categories). */
            if (!dead[b] && l->v[a].label == l->v[b].label &&
                box_iou(&l->v[a], &l->v[b]) > thresh)
                dead[b] = 1;
        }
    }

    /* Kept masks in score order, as torchvision returns them. */
    nv = G_malloc((l->n ? l->n : 1) * sizeof(struct sam_mask));
    for (i = 0; i < l->n; i++) {
        int a = o[i].idx;

        if (keep[a])
            nv[n++] = l->v[a];
        else
            G_free(l->v[a].m);
    }
    G_free(l->v);
    l->v = nv;
    l->cap = l->n ? l->n : 1;
    l->n = n;
    G_free(o);
    G_free(keep);
    G_free(dead);
}

/* Small region removal (remove_small_regions with cv2 8-connectivity).
 *
 * SAM2AutomaticMaskGenerator keeps this step as a method but no longer
 * calls it; its min_mask_region_area instead fills holes and removes
 * sprinkles in the low-resolution logits with a CUDA-only kernel that
 * is silently skipped elsewhere. The module keeps the original SAM
 * full-resolution cleaning, which does not depend on the device.
 *
 * The mask is only stored over its bounding box, but SAM runs this on
 * the full image, where background outside the box is one connected
 * area. A background component touching a box side that is not also an
 * image edge is therefore connected to that outer background and is
 * never a hole. */

/* Label the 8-connected components of cells where (m[i] != 0) == fg.
 * Returns the component count; sizes[k] and touch[k] (bit per open box
 * side touched) are filled for components k = 1..count. */
static int label_components(const unsigned char *m, int w, int h, int fg,
                            int open_sides, int *lab, long **sizes, int **touch)
{
    int n = 0, cap = 64, *stack;
    size_t i, total = (size_t)w * h;

    *sizes = G_malloc((cap + 1) * sizeof(long));
    *touch = G_malloc((cap + 1) * sizeof(int));
    memset(lab, 0, total * sizeof(int));
    stack = G_malloc(total * sizeof(int));

    for (i = 0; i < total; i++) {
        size_t sp = 0;

        if (lab[i] || (m[i] != 0) != fg)
            continue;
        if (++n > cap) {
            cap *= 2;
            *sizes = G_realloc(*sizes, (cap + 1) * sizeof(long));
            *touch = G_realloc(*touch, (cap + 1) * sizeof(int));
        }
        (*sizes)[n] = 0;
        (*touch)[n] = 0;
        lab[i] = n;
        stack[sp++] = (int)i;
        while (sp) {
            int c = stack[--sp], cx = c % w, cy = c / w, dx, dy;

            (*sizes)[n]++;
            if (cx == 0)
                (*touch)[n] |= open_sides & 1;
            if (cy == 0)
                (*touch)[n] |= open_sides & 2;
            if (cx == w - 1)
                (*touch)[n] |= open_sides & 4;
            if (cy == h - 1)
                (*touch)[n] |= open_sides & 8;
            for (dy = -1; dy <= 1; dy++) {
                for (dx = -1; dx <= 1; dx++) {
                    int nx = cx + dx, ny = cy + dy;
                    size_t ni;

                    if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                        continue;
                    ni = (size_t)ny * w + nx;
                    if (!lab[ni] && (m[ni] != 0) == fg) {
                        lab[ni] = n;
                        stack[sp++] = (int)ni;
                    }
                }
            }
        }
    }
    G_free(stack);
    return n;
}

/* Returns 1 if the mask changed. */
static int remove_small(unsigned char *m, int w, int h, int open_sides,
                        long thresh, int holes)
{
    size_t total = (size_t)w * h, i;
    int *lab = G_malloc(total * sizeof(int));
    long *sizes;
    int *touch, n, k, nsmall = 0;
    char *small;

    n = label_components(m, w, h, holes ? 0 : 1, open_sides, lab, &sizes,
                         &touch);
    small = G_calloc(n + 1, 1);
    for (k = 1; k <= n; k++) {
        if (sizes[k] < thresh && !(holes && touch[k])) {
            small[k] = 1;
            nsmall++;
        }
    }
    if (nsmall == 0) {
        G_free(lab);
        G_free(sizes);
        G_free(touch);
        G_free(small);
        return 0;
    }
    if (holes) {
        for (i = 0; i < total; i++)
            if (lab[i] && small[lab[i]])
                m[i] = 1;
    }
    else {
        if (nsmall == n) {
            /* Everything is small: keep the largest island. */
            int best = 1;

            for (k = 2; k <= n; k++)
                if (sizes[k] > sizes[best])
                    best = k;
            small[best] = 0;
        }
        for (i = 0; i < total; i++)
            if (lab[i] && small[lab[i]])
                m[i] = 0;
    }
    G_free(lab);
    G_free(sizes);
    G_free(touch);
    G_free(small);
    return 1;
}

/* Shrink a mask's stored box to its set cells; returns the area. */
static long tighten(struct sam_mask *mk)
{
    int w = mk->x1 - mk->x0 + 1, h = mk->y1 - mk->y0 + 1;
    int x, y, nx0 = w, ny0 = h, nx1 = -1, ny1 = -1;
    long area = 0;
    unsigned char *nm;

    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++)
            if (mk->m[(size_t)y * w + x]) {
                area++;
                if (x < nx0)
                    nx0 = x;
                if (x > nx1)
                    nx1 = x;
                if (y < ny0)
                    ny0 = y;
                if (y > ny1)
                    ny1 = y;
            }
    if (area == 0 || (nx0 == 0 && ny0 == 0 && nx1 == w - 1 && ny1 == h - 1))
        return area;
    nm = G_malloc((size_t)(nx1 - nx0 + 1) * (ny1 - ny0 + 1));
    for (y = ny0; y <= ny1; y++)
        memcpy(nm + (size_t)(y - ny0) * (nx1 - nx0 + 1),
               mk->m + (size_t)y * w + nx0, nx1 - nx0 + 1);
    G_free(mk->m);
    mk->m = nm;
    mk->x1 = mk->x0 + nx1;
    mk->y1 = mk->y0 + ny1;
    mk->x0 += nx0;
    mk->y0 += ny0;
    return area;
}

static void postprocess_small_regions(struct mask_list *l, int W, int H,
                                      long min_area, double nms_thresh)
{
    double *score = G_malloc((l->n ? l->n : 1) * sizeof(double));
    int i;

#pragma omp parallel for schedule(dynamic)
    for (i = 0; i < l->n; i++) {
        struct sam_mask *mk = &l->v[i];
        int w = mk->x1 - mk->x0 + 1, h = mk->y1 - mk->y0 + 1;
        int open = (mk->x0 > 0 ? 1 : 0) | (mk->y0 > 0 ? 2 : 0) |
                   (mk->x1 < W - 1 ? 4 : 0) | (mk->y1 < H - 1 ? 8 : 0);
        int changed;

        changed = remove_small(mk->m, w, h, open, min_area, 1);
        changed |= remove_small(mk->m, w, h, open, min_area, 0);
        score[i] = changed ? 0.0 : 1.0;
    }
    for (i = 0; i < l->n; i++)
        l->v[i].area = tighten(&l->v[i]);
    nms(l, score, NULL, nms_thresh);
    G_free(score);
}

/* Debug dumps: raw float32 files for comparison with the PyTorch
 * reference implementation (see tests/). */
static void dump_floats(const char *dir, const char *what, int crop,
                        const float *v, size_t n)
{
    char path[GPATH_MAX];
    FILE *fp;

    snprintf(path, sizeof(path), "%s/crop%03d_%s.f32", dir, crop, what);
    fp = fopen(path, "wb");
    if (!fp || fwrite(v, sizeof(float), n, fp) != n)
        G_fatal_error(_("Unable to write debug dump <%s>"), path);
    fclose(fp);
}

/* Keep postprocessed mask k of a crop unless it is empty or touches an
 * inner crop edge (is_box_near_crop_edge); st holds its statistics. */
static void accept_mask(struct sam_model *m, const struct sam_image *img,
                        const struct crop *c, int k, const int *st, float score,
                        int label, struct mask_list *cl)
{
    const int W = img->cols, H = img->rows;
    const int w = c->x1 - c->x0, h = c->y1 - c->y0;
    double bx[4], cb[4], ob[4] = {0, 0, W, H};
    struct sam_mask mk;
    int e;

    if (st[4] < 0)
        return; /* Empty mask. */
    mk.x0 = c->x0 + st[2];
    mk.y0 = c->y0 + st[3];
    mk.x1 = c->x0 + st[4];
    mk.y1 = c->y0 + st[5];
    bx[0] = mk.x0;
    bx[1] = mk.y0;
    bx[2] = mk.x1;
    bx[3] = mk.y1;
    cb[0] = c->x0;
    cb[1] = c->y0;
    cb[2] = c->x1;
    cb[3] = c->y1;
    for (e = 0; e < 4; e++)
        if (fabs(bx[e] - cb[e]) <= EDGE_ATOL &&
            !(fabs(bx[e] - ob[e]) <= EDGE_ATOL))
            return;
    mk.m = G_malloc((size_t)(st[4] - st[2] + 1) * (st[5] - st[3] + 1));
    sam_read_mask(m, k, h, w, st[2], st[3], st[4], st[5], mk.m);
    mk.iou = score;
    mk.stability = st[1] > 0 ? (float)st[0] / st[1] : 0.0f;
    mk.crop_score = 1.0 / ((double)w * h);
    mk.area = 0;
    mk.label = label;
    list_push(cl, &mk);
}

/* Search one crop for the text prompts (Sam3Processor.set_text_prompt:
 * every query whose score exceeds the threshold, no NMS). */
static void text_crop(struct sam_model *m, const struct sam_image *img,
                      const struct crop *c, int ci, const struct amg_params *p,
                      float *encbuf, struct mask_list *out)
{
    const int w = c->x1 - c->x0, h = c->y1 - c->y0;
    const int cap = text_mask_capacity(p->text);
    int qidx[TEXT_QUERIES], *sel = G_malloc(cap * sizeof(int));
    int *stats = G_malloc(6 * (size_t)cap * sizeof(int));
    float score[TEXT_QUERIES];
    int k, n, b, i, before = out->n;
    double t0 = now();

    progress_begin(PG_ENCODE, 1);
    image_encoder_input(img, c->x0, c->y0, w, h, encbuf);
    sam_encode_trunk(m, encbuf);
    text_set_image(p->text);
    clFinish(m->ocl->queue);
    progress_end();
    G_verbose_message(_("Image encoder: %.2f s"), now() - t0);
    if (p->dump_dir) {
        size_t ne = (size_t)SAM_GRID * SAM_GRID * SAM_DIM;
        float *feat = G_malloc(ne * sizeof(float));
        float *txt = G_malloc(TOK_CONTEXT * SAM_DIM * sizeof(float));
        char path[GPATH_MAX];
        FILE *fp;
        int y, ntok;

        snprintf(path, sizeof(path), "%s/crop%03d_rgb_%dx%d.u8", p->dump_dir,
                 ci, h, w);
        fp = fopen(path, "wb");
        if (!fp)
            G_fatal_error(_("Unable to write debug dump <%s>"), path);
        for (y = 0; y < h; y++)
            fwrite(img->rgb + ((size_t)(c->y0 + y) * img->cols + c->x0) * 3, 3,
                   w, fp);
        fclose(fp);
        dump_floats(p->dump_dir, "input", ci, encbuf, 3L * SAM_IMG * SAM_IMG);
        text_read_features(p->text, feat, txt, &ntok);
        dump_floats(p->dump_dir, "detector", ci, feat, ne);
        dump_floats(p->dump_dir, "text", ci, txt, (size_t)ntok * SAM_DIM);
        G_free(feat);
        G_free(txt);
    }

    for (k = 0; k < p->nprompts; k++) {
        progress_begin(PG_TEXT, 1);
        t0 = now();
        n = text_detect(p->text, k, (float)p->text_thresh, qidx, score);
        for (b = 0; b < n; b += cap) {
            int cnt = n - b < cap ? n - b : cap;

            for (i = 0; i < cnt; i++)
                sel[i] = i;
            text_masks(p->text, cnt, qidx + b);
            sam_postprocess(m, cnt, sel, h, w, (float)p->stability_offset, 0,
                            stats);
            for (i = 0; i < cnt; i++)
                accept_mask(m, img, c, i, stats + 6 * i, score[b + i], k, out);
        }
        G_verbose_message(_("Prompt %d: %d detections, %.2f s"), k + 1, n,
                          now() - t0);
    }
    progress_end();
    G_verbose_message(_("Crop %d: %d masks kept"), ci + 1, out->n - before);
    G_free(sel);
    G_free(stats);
}

/* Process one crop (_process_crop + _process_batch). */
static void process_crop(struct sam_model *m, const struct sam_image *img,
                         const struct crop *c, int ci,
                         const struct amg_params *p, float *encbuf,
                         struct mask_list *out)
{
    const int w = c->x1 - c->x0, h = c->y1 - c->y0;
    int n_side =
        (int)(p->points_per_side / pow(p->crop_points_downscale, c->layer));
    int npts, b, i, k;
    struct mask_list cl = {0, 0, NULL};
    float *pts, *iou;
    int *sel, *stats;
    double *iouv;

    double t0 = now(), t1;

    progress_begin(PG_ENCODE, 1);
    image_encoder_input(img, c->x0, c->y0, w, h, encbuf);
    sam_encode(m, encbuf);
    clFinish(m->ocl->queue);
    progress_end();
    t1 = now();
    G_verbose_message(_("Image encoder: %.2f s"), t1 - t0);
    if (p->dump_dir) {
        size_t ne = (size_t)SAM_GRID * SAM_GRID * SAM_DIM;
        size_t n0 = (size_t)SAM_LOWRES * SAM_LOWRES * (SAM_DIM / 8);
        size_t n1 = (size_t)SAM_LOWRES * SAM_LOWRES / 4 * (SAM_DIM / 4);
        float *emb = G_malloc(ne * sizeof(float));
        float *s0 = G_malloc(n0 * sizeof(float));
        float *s1 = G_malloc(n1 * sizeof(float));
        char path[GPATH_MAX];
        FILE *fp;
        int y;

        /* The 8-bit crop as HxWx3 bytes, for running the reference
         * mask generator on identical pixels. */
        snprintf(path, sizeof(path), "%s/crop%03d_rgb_%dx%d.u8", p->dump_dir,
                 ci, h, w);
        fp = fopen(path, "wb");
        if (!fp)
            G_fatal_error(_("Unable to write debug dump <%s>"), path);
        for (y = 0; y < h; y++)
            fwrite(img->rgb + ((size_t)(c->y0 + y) * img->cols + c->x0) * 3, 3,
                   w, fp);
        fclose(fp);

        dump_floats(p->dump_dir, "input", ci, encbuf, 3L * SAM_IMG * SAM_IMG);
        sam_read_features(m, emb, s0, s1);
        dump_floats(p->dump_dir, "embedding", ci, emb, ne);
        dump_floats(p->dump_dir, "feat_s0", ci, s0, n0);
        dump_floats(p->dump_dir, "feat_s1", ci, s1, n1);
        G_free(emb);
        G_free(s0);
        G_free(s1);
    }

    /* build_point_grid, scaled to the crop, then to the encoder input
     * (transform_coords): the crop is squashed to 1024 x 1024, so a
     * normalized grid point maps to the same encoder pixel in any crop. */
    if (n_side < 1)
        n_side = 1;
    npts = n_side * n_side;
    pts = G_malloc(2 * (size_t)npts * sizeof(float));
    for (i = 0; i < npts; i++) {
        double off = 1.0 / (2 * n_side);
        double gx = n_side > 1
                        ? off + (i % n_side) * (1.0 - 2 * off) / (n_side - 1)
                        : 0.5;
        double gy = n_side > 1
                        ? off + (i / n_side) * (1.0 - 2 * off) / (n_side - 1)
                        : 0.5;

        pts[2 * i] = (float)(gx * w / w * SAM_IMG);
        pts[2 * i + 1] = (float)(gy * h / h * SAM_IMG);
    }

    iou = G_malloc(3 * (size_t)m->batch * sizeof(float));
    sel = G_malloc(3 * (size_t)m->batch * sizeof(int));
    stats = G_malloc(18 * (size_t)m->batch * sizeof(int));

    for (b = 0; b < npts; b += m->batch) {
        int P = npts - b < m->batch ? npts - b : m->batch, nsel = 0;

        progress_begin(PG_DECODE, 1);
        sam_decode(m, P, pts + 2 * (size_t)b, iou);
        for (i = 0; i < 3 * P; i++)
            if (p->pred_iou_thresh <= 0.0 || iou[i] > p->pred_iou_thresh)
                sel[nsel++] = i;
        if (!nsel)
            continue;
        sam_postprocess(m, nsel, sel, h, w, (float)p->stability_offset, 1,
                        stats);

        for (k = 0; k < nsel; k++) {
            const int *st = stats + 6 * k;
            double stab = st[1] > 0 ? (double)st[0] / st[1] : 0.0;

            if (p->stability_thresh > 0.0 && !(stab >= p->stability_thresh))
                continue;
            accept_mask(m, img, c, k, st, iou[sel[k]], -1, &cl);
        }
    }
    progress_end();
    G_verbose_message(_("Mask decoder, %d prompts: %.2f s"), npts, now() - t1);

    iouv = G_malloc((cl.n ? cl.n : 1) * sizeof(double));
    for (i = 0; i < cl.n; i++)
        iouv[i] = cl.v[i].iou;
    nms(&cl, iouv, NULL, p->box_nms_thresh);
    G_free(iouv);

    for (i = 0; i < cl.n; i++)
        list_push(out, &cl.v[i]);
    G_verbose_message(_("Crop %d: %d masks kept"), ci + 1, cl.n);
    G_free(cl.v);
    G_free(pts);
    G_free(iou);
    G_free(sel);
    G_free(stats);
}

/* The crops of a W x H region: the crop layers of each of the nx x ny
 * tiles. */
static struct crop *region_crops(int W, int H, const struct amg_params *p,
                                 int *ncrops, int *nx, int *ny)
{
    int *xs, *ys, ix, iy, ccap = 0;
    struct crop *crops = NULL;

    *ncrops = 0;
    xs = G_malloc((W + 2) * sizeof(int));
    ys = G_malloc((H + 2) * sizeof(int));
    *nx = tile_starts(W, p->tile_size, p->tile_overlap, xs);
    *ny = tile_starts(H, p->tile_size, p->tile_overlap, ys);
    for (iy = 0; iy < *ny; iy++) {
        for (ix = 0; ix < *nx; ix++) {
            int tw = W < p->tile_size ? W : p->tile_size;
            int th = H < p->tile_size ? H : p->tile_size;

            crop_boxes(tw, th, p->crop_layers, p->crop_overlap_ratio, xs[ix],
                       ys[iy], &crops, ncrops, &ccap);
        }
    }
    G_free(xs);
    G_free(ys);
    return crops;
}

void amg_plan(int W, int H, const struct amg_params *p, int batch)
{
    int ncrops, nx, ny, i;
    struct crop *crops = region_crops(W, H, p, &ncrops, &nx, &ny);

    progress_plan(PG_ENCODE, ncrops);
    for (i = 0; i < ncrops; i++) {
        int n_side = (int)(p->points_per_side /
                           pow(p->crop_points_downscale, crops[i].layer));
        int npts = n_side < 1 ? 1 : n_side * n_side;

        if (p->nprompts > 0)
            progress_plan(PG_TEXT, p->nprompts);
        else
            progress_plan(PG_DECODE, (npts + batch - 1) / batch);
    }
    progress_plan(PG_CLEAN, 1);
    G_free(crops);
}

void amg_generate(struct sam_model *m, const struct sam_image *img,
                  const struct amg_params *p, struct mask_list *out)
{
    const int W = img->cols, H = img->rows;
    int nx, ny, i, ncrops;
    struct crop *crops;
    float *encbuf;

    memset(out, 0, sizeof(*out));
    crops = region_crops(W, H, p, &ncrops, &nx, &ny);
    if (nx * ny > 1)
        G_message(_("Region split into %d x %d tiles of %d cells"), nx, ny,
                  p->tile_size);

    encbuf = G_malloc(3L * SAM_IMG * SAM_IMG * sizeof(float));
    for (i = 0; i < ncrops; i++) {
        G_message(_("Segmenting crop %d of %d (%d x %d cells)..."), i + 1,
                  ncrops, crops[i].x1 - crops[i].x0, crops[i].y1 - crops[i].y0);
        if (p->nprompts > 0)
            text_crop(m, img, &crops[i], i, p, encbuf, out);
        else
            process_crop(m, img, &crops[i], i, p, encbuf, out);
    }
    G_free(encbuf);

    progress_begin(PG_CLEAN, 1);
    for (i = 0; i < out->n; i++) {
        struct sam_mask *mk = &out->v[i];
        size_t j, n = (size_t)(mk->x1 - mk->x0 + 1) * (mk->y1 - mk->y0 + 1);

        mk->area = 0;
        for (j = 0; j < n; j++)
            mk->area += mk->m[j];
    }

    if (ncrops > 1) {
        double *s1 = G_malloc((out->n ? out->n : 1) * sizeof(double));
        double *s2 = G_malloc((out->n ? out->n : 1) * sizeof(double));

        for (i = 0; i < out->n; i++) {
            s1[i] = out->v[i].crop_score;
            s2[i] = out->v[i].iou;
        }
        nms(out, s1, s2, p->crop_nms_thresh);
        G_free(s1);
        G_free(s2);
    }

    if (p->min_region_area > 0)
        postprocess_small_regions(out, W, H, p->min_region_area,
                                  p->box_nms_thresh > p->crop_nms_thresh
                                      ? p->box_nms_thresh
                                      : p->crop_nms_thresh);

    G_free(crops);
    progress_end();
    G_message(_("%d masks generated"), out->n);
}
