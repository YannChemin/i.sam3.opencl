## DESCRIPTION

*i.sam3.opencl* segments a 3-band imagery group with the Segment
Anything Model 3 (SAM 3) and writes the segments as a clean vector area
map. The network runs on an OpenCL device (GPU or CPU), with weights read
directly from the original PyTorch checkpoint file: neither Python nor
PyTorch is needed at run time.

The module works in one of two ways:

- **Automatic segmentation** (no **prompt**): SAM 3's point-prompt head
  is prompted with a regular grid of points, and the masks it returns
  are kept according to their predicted quality and stability,
  de-duplicated and cleaned. Everything visible is segmented.
- **Text prompts** (**prompt**): SAM 3's concept detector finds every
  instance of each short noun phrase given, such as *building* or
  *swimming pool*, and only those become segments.

In both cases the overlapping masks are then turned into a
non-overlapping, topologically clean vector map with one attribute row
per segment. SAM 3's video tracker is not used.

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

### Text prompts

Each **prompt** is a short English noun phrase in plain ASCII text
(several are given comma-separated). SAM 3 encodes it with its text
encoder, fuses it with the image features, and its detector proposes 200
candidate objects per crop. A candidate is kept when its score, the
model's probability that it matches the prompt multiplied by the
probability that the concept is present in the crop at all, exceeds
**threshold** (0.5 by default, as in SAM 3's own `Sam3Processor`); its
mask is the detector's segmentation. Crops are processed as in automatic
segmentation (**tile_size**, **tile_overlap**, **crop_layers**), masks
cut by an inner crop edge are discarded, and duplicates found in
overlapping crops are removed (box IoU above 0.7), separately for each
prompt. The options of the point-grid generator (**points_per_side**,
**iou_threshold**, **stability_threshold**, **nms_threshold**) are not
used.

Prompts are matched as SAM 3 was trained: on short, concrete noun
phrases. A segment belongs to the prompt that found it (the **prompt**
column); where masks of different prompts overlap, the **overlap** rule
decides, with *score* using the detection score.

### Mask generation

The region is processed in crops. A region no larger than **tile_size**
cells is one crop, a larger region is split into tiles of **tile_size**
cells overlapping by **tile_overlap** cells. With **crop_layers** greater
than 0, each tile is additionally segmented in zoomed-in sub-crops (4 in
layer 1, 16 in layer 2, ...), prompted with **points_per_side** divided
by **crop_points_downscale** per layer.

Each crop is resized to 1008 x 1008 pixels, regardless of its aspect
ratio, as SAM 3 does (the 8-bit antialiased bilinear resize of
torchvision, reproduced to the bit), and passed through the image
encoder. The mask
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

With text prompts, the table holds the columns **cat**, **mask**,
**prompt** (the text prompt that found the segment), **score** (its
detection score), **cells** and **area**.

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

The weights used take 1739 MiB of device memory (the checkpoint's
text encoder, detector and tracker memory are not loaded) and the work
buffers 2497 MiB with **batch**=32; a smaller **batch** reduces the mask
decoder's share. Buffers are kept below the device's maximum allocation
size.

### Device profile and precision

The **profile** chooses how buffers are allocated and which matrix
product kernels run, to suit the device:

- *cpu*: a device working in host memory, such as PoCL. Buffers are
  mapped instead of copied, and the matrix products use kernels without
  local memory that stream rows with 8-wide vector loads, which PoCL
  runs far better than the GPU kernels.
- *igpu*: a GPU sharing host memory (integrated graphics, APU). Transfers
  map the buffers instead of copying them; the driver places the
  buffers (on Intel, buffers allocated in host memory slowed the
  kernels by about 15 %).
- *dgpu*: a GPU with its own memory on internal PCI Express. Buffers
  live in device memory.
- *egpu*: a GPU behind a slow link, such as a Thunderbolt or USB4
  enclosure. Buffers live in device memory and large transfers go
  through a pinned staging buffer. The weights stay on the device for
  the whole run.

With **profile**=*auto* the profile is detected: *cpu* for a CPU
device, *igpu* for a GPU that reports unified memory, *egpu* for a GPU
whose PCI function Linux marks as removable (when the driver reports
its PCI address), *dgpu* otherwise. The chosen profile and the reason
are printed. A profile can be forced to compare them; every profile
gives the same results, only the speed differs.

**precision**=*fp16* stores the weight matrices of the linear layers
(most of the model's memory) as IEEE half-precision numbers, read with
the core OpenCL `vload_half` functions, so no half-precision extension
is needed; the arithmetic stays in single precision. It halves their
memory and the bytes read by the matrix products, which matters on
devices limited by memory bandwidth (integrated GPUs, CPUs). Rounding
the weights to half precision changes the masks slightly; check the
result against **precision**=*fp32* (the default) on a sample before
relying on it.

### Performance

On a CPU (Intel Core i7-1165G7, 4 cores, PoCL), the image encoder took
about 1020 s per crop and the mask decoder about 46 s per 64 prompts;
with text prompts, the concept detector and its masks took about 75 s per
prompt and crop. A GPU is expected to be one to two orders of magnitude
faster. The number of crops, and so the run time, grows with the region
size divided by **tile_size**, and with **crop_layers**.

### Choosing the resolution

SAM 3 sees every crop at 1008 x 1008 pixels. Objects should span at
least ten or so pixels at that scale: set the region resolution and
**tile_size** accordingly. A smaller **tile_size** zooms in (crops are
upsampled to 1008 pixels), which finds smaller objects at the cost of
more crops. Square tiles avoid distorting objects: a crop much longer
than wide is squashed to a square.

### Text encoder and tokenizer

The text encoder (1.2 GiB of weights) is loaded, run once for all the
prompts and released before the image is processed. Its tokenizer is a C
port of CLIP's byte-pair encoder, reading the vocabulary file installed
with the module (`bpe_simple_vocab_16e6.txt.gz`, from OpenAI CLIP, MIT
licence); prompts must be plain ASCII.

### Debugging aids

When the environment variable `I_SAM3_OPENCL_DUMP` names a directory,
the 8-bit crops, the encoder inputs, the image embeddings, the
high-resolution features (with text prompts: the detector features and
the first prompt's text features) and the list of masks are written
there. The scripts `tests/reference_compare.py` and
`tests/reference_compare_text.py` in the source tree compare them with
the PyTorch reference implementation.
`I_SAM3_OPENCL_PROFILE=1` prints the time spent in each OpenCL kernel. Each line gives the event time and the wall-clock time (some drivers, such as rusticl on Intel iris, report wrong event times), and for the matrix products the achieved GFLOP/s, which shows whether a kernel is limited by compute or by memory bandwidth.
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

### Text prompts

Find the buildings and the swimming pools of an orthophoto, keeping
detections with a score of at least 0.6:

```sh
i.group group=ortho_rgb input=ortho.red,ortho.green,ortho.blue
g.region raster=ortho.red -p
i.sam3.opencl group=ortho_rgb output=found \
    prompt="building,swimming pool" threshold=0.6
v.db.select found columns=cat,prompt,score,area | head
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
