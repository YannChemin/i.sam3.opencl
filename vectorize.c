/****************************************************************************
 *
 * MODULE:       i.sam3.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Conversion of overlapping SAM masks into a clean,
 *               topological vector area map: overlap resolution, small
 *               region merging, boundary tracing, simplification and
 *               topology cleaning.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <grass/dbmi.h>
#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>
#include <grass/vector.h>

#include "vectorize.h"

/* Overlap resolution: paint masks so that the winning mask of every
 * cell is painted last. */

struct paint_order {
    int idx;
    double k1, k2;
};

static int cmp_paint(const void *a, const void *b)
{
    const struct paint_order *x = a, *y = b;

    if (x->k1 != y->k1)
        return x->k1 < y->k1 ? -1 : 1;
    if (x->k2 != y->k2)
        return x->k2 < y->k2 ? -1 : 1;
    return x->idx - y->idx;
}

static int *paint_labels(const struct mask_list *l, const struct sam_image *img,
                         enum overlap_rule rule, long maxsize)
{
    const int cols = img->cols;
    struct paint_order *o = G_malloc((l->n ? l->n : 1) * sizeof(*o));
    int *lab = G_calloc((size_t)img->rows * cols, sizeof(int));
    int i, x, y, dropped = 0;

    for (i = 0; i < l->n; i++) {
        o[i].idx = i;
        switch (rule) {
        case OVERLAP_SMALL:
            o[i].k1 = -(double)l->v[i].area;
            o[i].k2 = l->v[i].iou;
            break;
        case OVERLAP_LARGE:
            o[i].k1 = (double)l->v[i].area;
            o[i].k2 = l->v[i].iou;
            break;
        default:
            o[i].k1 = l->v[i].iou;
            o[i].k2 = l->v[i].stability;
        }
    }
    qsort(o, l->n, sizeof(*o), cmp_paint);

    for (i = 0; i < l->n; i++) {
        const struct sam_mask *mk = &l->v[o[i].idx];
        int w = mk->x1 - mk->x0 + 1;

        if (maxsize > 0 && mk->area > maxsize) {
            dropped++;
            continue;
        }
        for (y = mk->y0; y <= mk->y1; y++)
            for (x = mk->x0; x <= mk->x1; x++) {
                size_t c = (size_t)y * cols + x;

                if (mk->m[(size_t)(y - mk->y0) * w + (x - mk->x0)] &&
                    img->valid[c])
                    lab[c] = o[i].idx + 1;
            }
    }
    if (dropped)
        G_message(_("%d masks larger than %ld cells discarded"), dropped,
                  maxsize);
    G_free(o);
    return lab;
}

/* Region labelling and small region merging. */

/* 4-connected components of equal key (key -1 marks NULL cells).
 * Returns the component count. */
static int label_regions(const int *key, int rows, int cols, int *comp,
                         int **ckey, long **csize)
{
    size_t n = (size_t)rows * cols, i;
    int *stack = G_malloc(n * sizeof(int));
    int nc = 0, cap = 1024;

    *ckey = G_malloc(cap * sizeof(int));
    *csize = G_malloc(cap * sizeof(long));
    for (i = 0; i < n; i++)
        comp[i] = -1;

    for (i = 0; i < n; i++) {
        size_t sp = 0;
        int k = key[i];

        if (comp[i] >= 0)
            continue;
        if (nc == cap) {
            cap *= 2;
            *ckey = G_realloc(*ckey, cap * sizeof(int));
            *csize = G_realloc(*csize, cap * sizeof(long));
        }
        (*ckey)[nc] = k;
        (*csize)[nc] = 0;
        comp[i] = nc;
        stack[sp++] = (int)i;
        while (sp) {
            int c = stack[--sp], r = c / cols, cc = c % cols;
            int nb[4], j;

            (*csize)[nc]++;
            nb[0] = cc > 0 ? c - 1 : -1;
            nb[1] = cc < cols - 1 ? c + 1 : -1;
            nb[2] = r > 0 ? c - cols : -1;
            nb[3] = r < rows - 1 ? c + cols : -1;
            for (j = 0; j < 4; j++)
                if (nb[j] >= 0 && comp[nb[j]] < 0 && key[nb[j]] == k) {
                    comp[nb[j]] = nc;
                    stack[sp++] = nb[j];
                }
        }
        nc++;
    }
    G_free(stack);
    return nc;
}

static int uf_find(int *parent, int a)
{
    while (parent[a] != a) {
        parent[a] = parent[parent[a]];
        a = parent[a];
    }
    return a;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;

    return x < y ? -1 : x > y;
}

struct by_size {
    int c;
    long s;
};

static int cmp_size(const void *a, const void *b)
{
    const struct by_size *x = a, *y = b;

    if (x->s != y->s)
        return x->s < y->s ? -1 : 1;
    return x->c - y->c;
}

/* Merge every component smaller than minsize into the neighbouring
 * component it shares the longest border with (NULL cells excluded),
 * smallest first. Returns the union-find parent array; ckey is updated
 * so that roots carry the key of the merged region. */
static int *merge_small(const int *comp, int rows, int cols, int nc, int *ckey,
                        const long *csize, long minsize)
{
    int *parent = G_malloc(nc * sizeof(int));
    int *next = G_malloc(nc * sizeof(int)), *tail = G_malloc(nc * sizeof(int));
    long *size = G_malloc(nc * sizeof(long)), *acc;
    int *adj_start, *adj_nb, *touched, ntouched, c, r;
    long *adj_cnt;
    uint64_t *pairs = NULL;
    size_t np = 0, pcap = 0, i, nu;
    struct by_size *order;
    int norder = 0;

    for (c = 0; c < nc; c++) {
        parent[c] = c;
        next[c] = -1;
        tail[c] = c;
        size[c] = csize[c];
    }
    if (minsize <= 1)
        goto done;

    /* Unique adjacent component pairs with their shared border length. */
    for (r = 0; r < rows; r++) {
        for (c = 0; c < cols; c++) {
            size_t i0 = (size_t)r * cols + c;
            int a = comp[i0], k;

            for (k = 0; k < 2; k++) {
                int b;

                if (k == 0 && c + 1 >= cols)
                    continue;
                if (k == 1 && r + 1 >= rows)
                    continue;
                b = comp[k == 0 ? i0 + 1 : i0 + cols];
                if (a == b)
                    continue;
                if (np == pcap) {
                    pcap = pcap ? 2 * pcap : 65536;
                    pairs = G_realloc(pairs, pcap * sizeof(uint64_t));
                }
                pairs[np++] = a < b ? ((uint64_t)a << 32) | (uint32_t)b
                                    : ((uint64_t)b << 32) | (uint32_t)a;
            }
        }
    }
    qsort(pairs, np, sizeof(uint64_t), cmp_u64);

    /* CSR adjacency, both directions. */
    adj_start = G_calloc(nc + 1, sizeof(int));
    for (i = 0; i < np; i++) {
        if (i > 0 && pairs[i] == pairs[i - 1])
            continue;
        adj_start[(int)(pairs[i] >> 32) + 1]++;
        adj_start[(int)(pairs[i] & 0xffffffffu) + 1]++;
    }
    for (c = 0; c < nc; c++)
        adj_start[c + 1] += adj_start[c];
    nu = adj_start[nc];
    adj_nb = G_malloc((nu ? nu : 1) * sizeof(int));
    adj_cnt = G_malloc((nu ? nu : 1) * sizeof(long));
    {
        int *fill = G_malloc(nc * sizeof(int));

        memcpy(fill, adj_start, nc * sizeof(int));
        for (i = 0; i < np;) {
            size_t j = i;
            int a = (int)(pairs[i] >> 32), b = (int)(pairs[i] & 0xffffffffu);

            while (j < np && pairs[j] == pairs[i])
                j++;
            adj_nb[fill[a]] = b;
            adj_cnt[fill[a]++] = (long)(j - i);
            adj_nb[fill[b]] = a;
            adj_cnt[fill[b]++] = (long)(j - i);
            i = j;
        }
        G_free(fill);
    }
    G_free(pairs);

    order = G_malloc(nc * sizeof(struct by_size));
    for (c = 0; c < nc; c++)
        if (csize[c] < minsize && ckey[c] != -1) {
            order[norder].c = c;
            order[norder++].s = csize[c];
        }
    qsort(order, norder, sizeof(struct by_size), cmp_size);

    acc = G_calloc(nc, sizeof(long));
    touched = G_malloc(nc * sizeof(int));
    for (i = 0; i < (size_t)norder; i++) {
        int root = uf_find(parent, order[i].c), best = -1, m, j;

        if (size[root] >= minsize)
            continue;
        ntouched = 0;
        for (m = root; m >= 0; m = next[m]) {
            for (j = adj_start[m]; j < adj_start[m + 1]; j++) {
                int nr = uf_find(parent, adj_nb[j]);

                if (nr == root || ckey[nr] == -1)
                    continue;
                if (!acc[nr])
                    touched[ntouched++] = nr;
                acc[nr] += adj_cnt[j];
            }
        }
        for (j = 0; j < ntouched; j++) {
            int t = touched[j];

            if (best < 0 || acc[t] > acc[best] ||
                (acc[t] == acc[best] && size[t] > size[best]))
                best = t;
        }
        for (j = 0; j < ntouched; j++)
            acc[touched[j]] = 0;

        if (best < 0) {
            /* Enclosed by NULL cells only: drop it as unlabelled. */
            ckey[root] = 0;
            continue;
        }
        parent[root] = best;
        size[best] += size[root];
        next[tail[best]] = root;
        tail[best] = tail[root];
    }
    G_free(acc);
    G_free(touched);
    G_free(order);
    G_free(adj_start);
    G_free(adj_nb);
    G_free(adj_cnt);

done:
    G_free(next);
    G_free(tail);
    G_free(size);
    return parent;
}

/* Per-segment information, indexed by category. */
struct segment {
    int mask; /* 1-based SAM mask number. */
    long cells;
    int best_cell; /* Pole of inaccessibility (chamfer distance). */
    int best_dist;
};

/* Chamfer (3-4) distance to the segment border; picks for every
 * segment the cell farthest inside it, a robust centroid location that
 * survives boundary simplification. */
static void find_poles(const int *K, int rows, int cols, struct segment *seg)
{
    size_t n = (size_t)rows * cols, i;
    int *D = G_malloc(n * sizeof(int)), r, c;

    for (r = 0; r < rows; r++)
        for (c = 0; c < cols; c++) {
            size_t i0 = (size_t)r * cols + c;
            int k = K[i0], border;

            border = r == 0 || c == 0 || r == rows - 1 || c == cols - 1 ||
                     K[i0 - 1] != k || K[i0 + 1] != k || K[i0 - cols] != k ||
                     K[i0 + cols] != k;
            D[i0] = border ? 3 : INT32_MAX / 2;
        }

#define RELAX(nr, nc, w)                                                       \
    do {                                                                       \
        if ((nr) >= 0 && (nr) < rows && (nc) >= 0 && (nc) < cols) {            \
            size_t j = (size_t)(nr) * cols + (nc);                             \
            if (K[j] == K[i0] && D[j] + (w) < D[i0])                           \
                D[i0] = D[j] + (w);                                            \
        }                                                                      \
    } while (0)

    for (r = 0; r < rows; r++)
        for (c = 0; c < cols; c++) {
            size_t i0 = (size_t)r * cols + c;

            RELAX(r, c - 1, 3);
            RELAX(r - 1, c, 3);
            RELAX(r - 1, c - 1, 4);
            RELAX(r - 1, c + 1, 4);
        }
    for (r = rows - 1; r >= 0; r--)
        for (c = cols - 1; c >= 0; c--) {
            size_t i0 = (size_t)r * cols + c;

            RELAX(r, c + 1, 3);
            RELAX(r + 1, c, 3);
            RELAX(r + 1, c + 1, 4);
            RELAX(r + 1, c - 1, 4);
        }
#undef RELAX

    for (i = 0; i < n; i++) {
        int k = K[i];

        if (k > 0 && D[i] > seg[k].best_dist) {
            seg[k].best_dist = D[i];
            seg[k].best_cell = (int)i;
        }
    }
    G_free(D);
}

/* Boundary tracing on the cell-corner lattice. Vertex (r, c) is the
 * corner at row line r, column line c. Directions: 0 east, 1 south,
 * 2 west, 3 north. An edge exists where the categories on its two sides
 * differ (outside the region counts as 0). */

struct tracer {
    const int *K;
    int rows, cols;
    unsigned char *hvis, *vvis;
};

static int key_at(const struct tracer *t, int r, int c)
{
    if (r < 0 || c < 0 || r >= t->rows || c >= t->cols)
        return 0;
    return t->K[(size_t)r * t->cols + c];
}

/* Returns 1 if the edge leaving (r, c) in direction d exists, and
 * points *vis at its visited flag. */
static int edge(const struct tracer *t, int r, int c, int d,
                unsigned char **vis)
{
    switch (d) {
    case 0:
        if (c >= t->cols)
            return 0;
        *vis = t->hvis + (size_t)r * t->cols + c;
        return key_at(t, r - 1, c) != key_at(t, r, c);
    case 2:
        if (c <= 0)
            return 0;
        *vis = t->hvis + (size_t)r * t->cols + c - 1;
        return key_at(t, r - 1, c - 1) != key_at(t, r, c - 1);
    case 1:
        if (r >= t->rows)
            return 0;
        *vis = t->vvis + (size_t)r * (t->cols + 1) + c;
        return key_at(t, r, c - 1) != key_at(t, r, c);
    default:
        if (r <= 0)
            return 0;
        *vis = t->vvis + (size_t)(r - 1) * (t->cols + 1) + c;
        return key_at(t, r - 1, c - 1) != key_at(t, r - 1, c);
    }
}

static int degree(const struct tracer *t, int r, int c)
{
    unsigned char *v;
    int d, n = 0;

    for (d = 0; d < 4; d++)
        n += edge(t, r, c, d, &v);
    return n;
}

static const int DR[4] = {0, 1, 0, -1}, DC[4] = {1, 0, -1, 0};

struct chain {
    int n, cap;
    int *rc; /* Vertex row, column pairs. */
};

static void chain_add(struct chain *ch, int r, int c)
{
    if (ch->n == ch->cap) {
        ch->cap = ch->cap ? 2 * ch->cap : 64;
        ch->rc = G_realloc(ch->rc, 2 * ch->cap * sizeof(int));
    }
    ch->rc[2 * ch->n] = r;
    ch->rc[2 * ch->n + 1] = c;
    ch->n++;
}

/* Follow edges from (r0, c0) heading d until a node (degree != 2) or
 * back to the start, keeping only the turning vertices. */
static void walk(const struct tracer *t, int r0, int c0, int d,
                 struct chain *ch)
{
    int r = r0, c = c0;
    unsigned char *vis;

    ch->n = 0;
    chain_add(ch, r, c);
    for (;;) {
        int nd = -1, k;

        edge(t, r, c, d, &vis);
        *vis = 1;
        r += DR[d];
        c += DC[d];
        if ((r == r0 && c == c0) || degree(t, r, c) != 2) {
            chain_add(ch, r, c);
            return;
        }
        for (k = 0; k < 4; k++) {
            if (k == (d + 2) % 4)
                continue;
            if (edge(t, r, c, k, &vis) && !*vis) {
                nd = k;
                break;
            }
        }
        if (nd < 0) {
            chain_add(ch, r, c);
            return;
        }
        if (nd != d)
            chain_add(ch, r, c);
        d = nd;
    }
}

/* Douglas-Peucker simplification of a polyline in map coordinates. */

static double seg_dist2(const double *p, const double *a, const double *b)
{
    double dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy, u;
    double ex, ey;

    u = l2 > 0 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2 : 0.0;
    if (u < 0)
        u = 0;
    else if (u > 1)
        u = 1;
    ex = a[0] + u * dx - p[0];
    ey = a[1] + u * dy - p[1];
    return ex * ex + ey * ey;
}

static void dp_rec(const double *xy, int a, int b, double tol2, char *keep)
{
    double dmax = 0.0;
    int i, imax = -1;

    for (i = a + 1; i < b; i++) {
        double d = seg_dist2(xy + 2 * i, xy + 2 * a, xy + 2 * b);

        if (d > dmax) {
            dmax = d;
            imax = i;
        }
    }
    if (imax >= 0 && dmax > tol2) {
        keep[imax] = 1;
        dp_rec(xy, a, imax, tol2, keep);
        dp_rec(xy, imax, b, tol2, keep);
    }
}

/* Simplify xy in place; returns the new point count. Closed rings are
 * split at their farthest vertex and never reduced below a triangle. */
static int simplify(double *xy, int n, double tol)
{
    char *keep;
    int i, m = 0, closed;

    if (tol <= 0 || n <= 2)
        return n;
    closed = xy[0] == xy[2 * (n - 1)] && xy[1] == xy[2 * (n - 1) + 1];
    if (closed && n < 5)
        return n;

    keep = G_calloc(n, 1);
    keep[0] = keep[n - 1] = 1;
    if (closed) {
        double dmax = -1.0;
        int f = 1;

        for (i = 1; i < n - 1; i++) {
            double dx = xy[2 * i] - xy[0], dy = xy[2 * i + 1] - xy[1];

            if (dx * dx + dy * dy > dmax) {
                dmax = dx * dx + dy * dy;
                f = i;
            }
        }
        keep[f] = 1;
        dp_rec(xy, 0, f, tol * tol, keep);
        dp_rec(xy, f, n - 1, tol * tol, keep);
    }
    else
        dp_rec(xy, 0, n - 1, tol * tol, keep);

    for (i = 0; i < n; i++)
        m += keep[i];
    if (closed && m < 4) {
        G_free(keep);
        return n;
    }
    for (i = 0, m = 0; i < n; i++)
        if (keep[i]) {
            xy[2 * m] = xy[2 * i];
            xy[2 * m + 1] = xy[2 * i + 1];
            m++;
        }
    G_free(keep);
    return m;
}

static void write_boundaries(struct Map_info *Map, const int *K, int rows,
                             int cols, const struct Cell_head *win, double tol)
{
    struct tracer t = {K, rows, cols, NULL, NULL};
    struct chain ch = {0, 0, NULL};
    struct line_pnts *Points = Vect_new_line_struct();
    struct line_cats *Cats = Vect_new_cats_struct();
    double *xy = NULL;
    int xycap = 0, r, c, d, nlines = 0;
    unsigned char *vis;

    t.hvis = G_calloc((size_t)(rows + 1) * cols, 1);
    t.vvis = G_calloc((size_t)rows * (cols + 1), 1);

#define EMIT()                                                                 \
    do {                                                                       \
        int k_, n_;                                                            \
        if (ch.n > xycap) {                                                    \
            xycap = ch.n;                                                      \
            xy = G_realloc(xy, 2 * xycap * sizeof(double));                    \
        }                                                                      \
        for (k_ = 0; k_ < ch.n; k_++) {                                        \
            xy[2 * k_] = win->west + ch.rc[2 * k_ + 1] * win->ew_res;          \
            xy[2 * k_ + 1] = win->north - ch.rc[2 * k_] * win->ns_res;         \
        }                                                                      \
        n_ = simplify(xy, ch.n, tol);                                          \
        Vect_reset_line(Points);                                               \
        for (k_ = 0; k_ < n_; k_++)                                            \
            Vect_append_point(Points, xy[2 * k_], xy[2 * k_ + 1], 0.0);        \
        Vect_write_line(Map, GV_BOUNDARY, Points, Cats);                       \
        nlines++;                                                              \
    } while (0)

    /* Chains between nodes. */
    for (r = 0; r <= rows; r++) {
        G_percent(r, rows + 1, 10);
        for (c = 0; c <= cols; c++) {
            int deg = degree(&t, r, c);

            if (deg == 0 || deg == 2)
                continue;
            for (d = 0; d < 4; d++) {
                if (edge(&t, r, c, d, &vis) && !*vis) {
                    walk(&t, r, c, d, &ch);
                    EMIT();
                }
            }
        }
    }
    G_percent(1, 1, 1);

    /* Closed rings without any node. Every ring has a horizontal edge. */
    for (r = 0; r <= rows; r++)
        for (c = 0; c < cols; c++)
            if (edge(&t, r, c, 0, &vis) && !*vis) {
                walk(&t, r, c, 0, &ch);
                EMIT();
            }
#undef EMIT

    G_verbose_message(_("%d boundaries traced"), nlines);
    G_free(xy);
    G_free(ch.rc);
    G_free(t.hvis);
    G_free(t.vvis);
    Vect_destroy_line_struct(Points);
    Vect_destroy_cats_struct(Cats);
}

static void write_label_raster(const char *name, const int *K, int rows,
                               int cols, const char *source)
{
    struct History hist;
    CELL *buf = Rast_allocate_c_buf();
    int fd = Rast_open_new(name, CELL_TYPE), r, c;

    for (r = 0; r < rows; r++) {
        for (c = 0; c < cols; c++) {
            int k = K[(size_t)r * cols + c];

            if (k > 0)
                buf[c] = k;
            else
                Rast_set_c_null_value(&buf[c], 1);
        }
        Rast_put_c_row(fd, buf);
    }
    Rast_close(fd);
    G_free(buf);

    Rast_short_history(name, "raster", &hist);
    Rast_set_history(&hist, HIST_DATSRC_1, source);
    Rast_command_history(&hist);
    Rast_write_history(name, &hist);
}

static void write_attributes(struct Map_info *Map, const struct segment *seg,
                             const struct mask_list *masks, double cellarea)
{
    struct field_info *Fi;
    dbDriver *driver;
    dbString sql;
    struct line_pnts *Points = Vect_new_line_struct();
    struct line_cats *Cats = Vect_new_cats_struct();
    char buf[1024];
    int line, nlines, cat, area;

    Fi = Vect_default_field_info(Map, 1, NULL, GV_1TABLE);
    Vect_map_add_dblink(Map, 1, NULL, Fi->table, GV_KEY_COLUMN, Fi->database,
                        Fi->driver);
    driver = db_start_driver_open_database(Fi->driver,
                                           Vect_subst_var(Fi->database, Map));
    if (!driver)
        G_fatal_error(_("Unable to open database <%s> by driver <%s>"),
                      Fi->database, Fi->driver);
    db_set_error_handler_driver(driver);

    db_init_string(&sql);
    snprintf(buf, sizeof(buf),
             "create table %s (%s integer, mask integer, "
             "predicted_iou double precision, stability double precision, "
             "cells integer, area double precision)",
             Fi->table, GV_KEY_COLUMN);
    db_set_string(&sql, buf);
    if (db_execute_immediate(driver, &sql) != DB_OK)
        G_fatal_error(_("Unable to create table: %s"), db_get_string(&sql));
    if (db_create_index2(driver, Fi->table, GV_KEY_COLUMN) != DB_OK)
        G_warning(_("Unable to create index"));
    if (db_grant_on_table(driver, Fi->table, DB_PRIV_SELECT,
                          DB_GROUP | DB_PUBLIC) != DB_OK)
        G_fatal_error(_("Unable to grant privileges on table <%s>"), Fi->table);

    db_begin_transaction(driver);
    nlines = Vect_get_num_lines(Map);
    for (line = 1; line <= nlines; line++) {
        const struct sam_mask *mk;

        if (!Vect_line_alive(Map, line) ||
            Vect_read_line(Map, Points, Cats, line) != GV_CENTROID)
            continue;
        if (!Vect_cat_get(Cats, 1, &cat) || cat <= 0)
            continue;
        area = Vect_get_centroid_area(Map, line);
        mk = &masks->v[seg[cat].mask - 1];
        snprintf(buf, sizeof(buf),
                 "insert into %s values (%d, %d, %.6f, %.6f, %ld, %.6f)",
                 Fi->table, cat, seg[cat].mask, mk->iou, mk->stability,
                 seg[cat].cells,
                 area > 0 ? Vect_get_area_area(Map, area)
                          : seg[cat].cells * cellarea);
        db_set_string(&sql, buf);
        if (db_execute_immediate(driver, &sql) != DB_OK)
            G_fatal_error(_("Unable to insert new record: %s"),
                          db_get_string(&sql));
    }
    db_commit_transaction(driver);
    db_close_database_shutdown_driver(driver);
    db_free_string(&sql);
    Vect_destroy_line_struct(Points);
    Vect_destroy_cats_struct(Cats);
}

void vectorize_masks(const struct mask_list *masks, const struct sam_image *img,
                     const struct vect_params *vp)
{
    const int rows = img->rows, cols = img->cols;
    const size_t n = (size_t)rows * cols;
    struct Cell_head win;
    struct Map_info Map;
    struct line_pnts *Points;
    struct line_cats *Cats;
    struct segment *seg;
    int *lab, *comp, *ckey, *parent, *catof, *K, nc, ncat = 0, c, line;
    int nlines, removed_centroids = 0;
    long *csize;
    size_t i;
    double cellarea, removed_area = 0.0;

    G_get_window(&win);
    cellarea = win.ew_res * win.ns_res;

    G_message(_("Resolving mask overlaps..."));
    lab = paint_labels(masks, img, vp->overlap, vp->maxsize);
    for (i = 0; i < n; i++)
        if (!img->valid[i])
            lab[i] = -1;

    comp = G_malloc(n * sizeof(int));
    nc = label_regions(lab, rows, cols, comp, &ckey, &csize);
    G_verbose_message(_("%d connected pieces before cleaning"), nc);

    G_message(_("Merging pieces smaller than %ld cells..."), vp->minsize);
    parent = merge_small(comp, rows, cols, nc, ckey, csize, vp->minsize);

    /* One category per final region carrying a mask label. */
    catof = G_calloc(nc, sizeof(int));
    for (c = 0; c < nc; c++)
        if (uf_find(parent, c) == c && ckey[c] > 0)
            catof[c] = ++ncat;
    seg = G_calloc(ncat + 1, sizeof(struct segment));
    K = lab; /* Reuse: per-cell category, 0 = no segment. */
    for (i = 0; i < n; i++) {
        int root = uf_find(parent, comp[i]), k = catof[root];

        K[i] = k;
        if (k > 0) {
            seg[k].mask = ckey[root];
            seg[k].cells++;
        }
    }
    G_free(comp);
    G_free(ckey);
    G_free(csize);
    G_free(parent);
    G_free(catof);
    G_message(_("%d segments after cleaning"), ncat);
    if (ncat == 0)
        G_warning(_("No segment left: the output vector map will be empty"));

    if (vp->raster_out)
        write_label_raster(vp->raster_out, K, rows, cols, vp->source);

    for (c = 1; c <= ncat; c++)
        seg[c].best_dist = -1;
    find_poles(K, rows, cols, seg);

    if (Vect_open_new(&Map, vp->vector_out, WITHOUT_Z) < 0)
        G_fatal_error(_("Unable to create vector map <%s>"), vp->vector_out);
    Vect_hist_command(&Map);

    G_message(_("Tracing boundaries..."));
    write_boundaries(&Map, K, rows, cols, &win, vp->simplify);

    Points = Vect_new_line_struct();
    Cats = Vect_new_cats_struct();
    for (c = 1; c <= ncat; c++) {
        int cell = seg[c].best_cell;

        Vect_reset_line(Points);
        Vect_reset_cats(Cats);
        Vect_append_point(Points, win.west + (cell % cols + 0.5) * win.ew_res,
                          win.north - (cell / cols + 0.5) * win.ns_res, 0.0);
        Vect_cat_set(Cats, 1, c);
        Vect_write_line(&Map, GV_CENTROID, Points, Cats);
    }

    /* Topology cleaning: simplification may create crossings, pseudo
     * nodes and slivers. */
    G_message(_("Cleaning topology..."));
    Vect_build_partial(&Map, GV_BUILD_BASE);
    Vect_break_lines(&Map, GV_BOUNDARY, NULL);
    Vect_remove_duplicates(&Map, GV_BOUNDARY, NULL);
    Vect_merge_lines(&Map, GV_BOUNDARY, NULL, NULL);
    Vect_build_partial(&Map, GV_BUILD_NONE);
    Vect_build_partial(&Map, GV_BUILD_CENTROIDS);

    nlines = Vect_get_num_lines(&Map);
    for (line = 1; line <= nlines; line++) {
        if (!Vect_line_alive(&Map, line) ||
            Vect_get_line_type(&Map, line) != GV_CENTROID)
            continue;
        if (Vect_get_centroid_area(&Map, line) <= 0) {
            Vect_delete_line(&Map, line);
            removed_centroids++;
        }
    }
    if (removed_centroids)
        G_verbose_message(_("%d centroids fell outside or duplicated an area "
                            "after simplification and were removed"),
                          removed_centroids);
    Vect_remove_small_areas(&Map, cellarea, NULL, &removed_area);
    Vect_build_partial(&Map, GV_BUILD_NONE);
    Vect_build(&Map);

    write_attributes(&Map, seg, masks, cellarea);
    Vect_close(&Map);

    Vect_destroy_line_struct(Points);
    Vect_destroy_cats_struct(Cats);
    G_free(seg);
    G_free(K);
}
