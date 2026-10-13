## DESCRIPTION

*i.sam3.opencl* segments a 3-band imagery group with the Segment
Anything Model 3 (SAM 3) and writes the segments as a clean vector area
map. It runs SAM 3's point-prompt segmentation, its Perception Encoder
image backbone, the neck of its point-prompt head, the prompt encoder and
the mask decoder, on an OpenCL device (GPU or CPU), with weights read
directly from the original PyTorch checkpoint file: neither Python nor
PyTorch is needed at run time.

The module implements automatic mask generation with that point-prompt
head: the model is prompted with a regular grid of points, and the masks
it returns are kept according to their predicted quality and stability,
de-duplicated and cleaned. The overlapping masks are then turned into a
non-overlapping, topologically clean vector map with one attribute row
per segment. SAM 3's text-prompted concept detection and its video
tracker are not used.

The **group** (or its **subgroup**) must contain exactly three raster
maps, which are used as the red, green and blue channels in the order
they are listed in the group (see *i.group*). Each band is stretched
linearly between its lower and upper **percentile** to the 0-255 range
SAM 3 was trained on. Cells that are NULL in any band are not segmented.
Processing is done in the current computational region.

### Checkpoint

The **checkpoint** is the original `sam3.pt` (3.2 GiB) published by Meta
on Hugging Face (<https://huggingface.co/facebook/sam3>). The model is
gated: an account must request access there, and downloading needs that
account's access token.

Without a **checkpoint** file, the module looks for `sam3.pt` in the
weights folder: **weights_dir**, by default the central folder
`$SAM_MODELS_DIR/sam3`, or `~/models/sam/sam3` when the environment
variable `SAM_MODELS_DIR` is not set. If it is not there, it is
downloaded into that folder (created if needed) on first use and reused
afterwards. The download uses the token in the environment variable
`HF_TOKEN`, else the one saved by `hf auth login`
(`~/.cache/huggingface/token`); it is checked against the published size,
resumes where an interrupted one stopped, and waits for another run
downloading the same file. The **-n** flag forbids downloading: a missing
checkpoint is then an error. Checkpoints of SAM and SAM 2 are refused:
they belong to *i.sam.opencl* and *i.sam2.opencl*.

### Mask generation

The region is processed in crops. A region no larger than **tile_size**
cells is one crop, a larger region is split into tiles of **tile_size**
cells overlapping by **tile_overlap** cells. With **crop_layers** greater
than 0, each tile is additionally segmented in zoomed-in sub-crops (4 in
layer 1, 16 in layer 2, ...), prompted with **points_per_side** divided
by **crop_points_downscale** per layer.

Each crop is resized to 1008 x 1008 pixels, regardless of its aspect
ratio, as SAM 3 does, and passed through the image encoder. The mask
decoder is then prompted with a grid of **points_per_side** x
**points_per_side** points, **batch** prompts at a time, returning three
candidate masks per point. A mask is kept if its predicted IoU exceeds
**iou_threshold** and its stability score (the IoU of the mask
thresholded at plus and minus **stability_offset**) is at least
**stability_threshold**. Duplicates whose bounding boxes overlap by more
than **nms_threshold** (IoU) are suppressed, keeping the best one. Masks
touching an inner tile or crop edge are discarded, since the neighbouring
tile sees them whole. Holes and islands smaller than **minsize** cells
are finally removed from each mask.

As in SAM 3's point predictor, holes of at most 256 pixels in each
288 x 288 low-resolution mask are filled before the mask is resized to
the crop. The other steps and their defaults are those of SAM 2's
`SAM2AutomaticMaskGenerator` (SAM 3 has no generator of its own), plus
the full-resolution removal of holes and islands smaller than
**minsize** cells. With **minsize**=0, the module returns the same masks
as those steps run on the PyTorch SAM 3 predictor.

### Vectorization and cleaning

SAM 3 masks overlap: a field and the parcels inside it can both be
masks. The **overlap** rule chooses which mask a cell belongs to: the
smallest covering mask (*small*, finest partition, the default), the
largest (*large*) or the one with the highest predicted IoU (*score*).
Masks larger than **maxsize** cells are discarded first; this is the way
to drop the single mask SAM 3 often returns for the whole background of
a scene.

The resulting label raster is cleaned: every connected piece smaller
than **minsize** cells, including small unsegmented gaps, is merged into
the neighbouring piece it shares the longest border with. Boundaries are
traced along cell edges into shared topological boundaries, simplified
with the Douglas-Peucker algorithm using the **simplify** tolerance (one
cell by default; 0 keeps the raster staircase), and the topology is
cleaned (boundaries broken at intersections, duplicates removed,
pseudo-nodes merged, slivers smaller than one cell removed). Each segment
gets one centroid, placed at the cell farthest from its border.

Areas without a centroid are gaps that no mask covers. The optional
**raster** output holds the segment categories of the cleaned label
raster before simplification.

### Attribute table

| Column        | Description                                        |
| ------------- | -------------------------------------------------- |
| cat           | Segment category                                   |
| mask          | Number of the SAM 3 mask the segment comes from    |
| predicted_iou | Mask quality predicted by SAM 3                    |
| stability     | Stability score of the mask                        |
| cells         | Segment size in cells                              |
| area          | Area of the simplified polygon in square map units |

A mask split into several pieces by overlapping masks yields several
segments with the same **mask** number.

## NOTES

### Differences from SAM and SAM 2

SAM 3's image encoder is a 32-block vision transformer (Perception
Encoder, 1024 channels, 2D rotary position encoding, windowed attention
with four global blocks) trained jointly for concept detection and
tracking. Its point-prompt head is SAM 2's mask decoder on a 72 x 72
embedding of a 1008-pixel input, fed with 288 and 144 pixel
high-resolution features from a SimpleFPN neck. The encoder needs about
as much computation per crop as SAM's ViT-H, much more than SAM 2's.

### OpenCL device

With **device**=*auto*, the first GPU is used, else the first CPU
device. When several OpenCL platforms expose the same device (for
example Mesa's Clover and rusticl drivers for AMD GPUs), the first
platform on which the kernels compile is used; **platform** restricts
the choice to platforms whose name contains the given text. Mesa's
rusticl only exposes a GPU when enabled in the environment, e.g.
`RUSTICL_ENABLE=radeonsi`. A CPU OpenCL implementation such as PoCL
gives the same result, much more slowly.

The weights used take @WEIGHTS@ MiB of device memory (the checkpoint's
text encoder, detector and tracker memory are not loaded) and the work
buffers @WORK@ MiB with **batch**=32; a smaller **batch** reduces the mask
decoder's share. Buffers are kept below the device's maximum allocation
size.

### Performance

On a CPU (Intel Core i7-1165G7, 4 cores, PoCL), the image encoder took
@ENC@ s per crop and the mask decoder @DEC@ s per 64 prompts. A GPU is
expected to be one to two orders of magnitude faster. The number of
crops, and so the run time, grows with the region size divided by
**tile_size**, and with **crop_layers**.

### Choosing the resolution

SAM 3 sees every crop at 1008 x 1008 pixels. Objects should span at
least ten or so pixels at that scale: set the region resolution and
**tile_size** accordingly. A smaller **tile_size** zooms in (crops are
upsampled to 1008 pixels), which finds smaller objects at the cost of
more crops. Square tiles avoid distorting objects: a crop much longer
than wide is squashed to a square.

### Debugging aids

When the environment variable `I_SAM3_OPENCL_DUMP` names a directory,
the 8-bit crops, the encoder inputs, the image embeddings, the
high-resolution features and the list of masks are written there. The
script `tests/reference_compare.py` in the source tree compares them
with the PyTorch reference implementation.
`I_SAM3_OPENCL_PROFILE=1` prints the time spent in each OpenCL kernel.
`I_SAM3_OPENCL_DOWNLOAD_URL` replaces the download location
(`https://huggingface.co/facebook/sam3/resolve/main`), for a local
mirror.

## EXAMPLES

### Landsat scene (North Carolina sample dataset)

Segment a true-colour composite of Landsat 7 bands 3, 2 and 1, dropping
the background mask (the region has 475 x 527 cells); `sam3.pt` is taken
from, or downloaded into, `~/models/sam/sam3`:

```sh
g.region raster=lsat7_2002_30 -p
i.group group=lsat_rgb input=lsat7_2002_30,lsat7_2002_20,lsat7_2002_10
i.sam3.opencl group=lsat_rgb output=lsat_segments maxsize=50000
v.db.select lsat_segments | head
```

### Agricultural parcels

Delineate fields in a 2.5 m VNIR image with bands 1-3 grouped as RGB,
keeping only segments of at least 0.1 ha (160 cells) and a label raster
alongside the vector map:

```sh
i.group group=vnir_rgb input=vnir.1,vnir.2,vnir.3
g.region raster=vnir.1 -p
i.sam3.opencl group=vnir_rgb output=fields raster=fields minsize=160
```

## REFERENCES

Carion, N., Gustafson, L., Hu, Y.-T., Debnath, S., Hu, R., Suris, D., et
al., 2025. SAM 3: Segment Anything with Concepts.
[arXiv:2511.16719](https://arxiv.org/abs/2511.16719)

## SEE ALSO

*[i.group](i.group.md), [i.sam.opencl](i.sam.opencl.md),
[i.sam2.opencl](i.sam2.opencl.md),
[i.segment](i.segment.md), [r.to.vect](r.to.vect.md),
[v.clean](v.clean.md), [v.generalize](v.generalize.md)*

## AUTHORS

Yann Chemin
