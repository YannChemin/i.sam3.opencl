# i.sam3.opencl

GRASS GIS module that segments a 3-band imagery group with Meta's Segment
Anything Model 3 (SAM 3) on an OpenCL device (GPU or CPU) and writes the
segments as a clean vector area map. It runs SAM 3's point-prompt
segmentation (the Perception Encoder backbone, the neck of the
point-prompt head, the prompt encoder and the mask decoder) entirely in
OpenCL and reads the original PyTorch checkpoint directly: neither Python
nor PyTorch is needed at run time. See `i.sam3.opencl.md` for the full
manual.

This is the SAM 3 counterpart of *i.sam.opencl* (SAM) and *i.sam2.opencl*
(SAM 2); the three modules are independent. SAM 3's text-prompted concept
segmentation is not implemented yet.

## SAM 3 weights

The weights are not shipped with the module: `sam3.pt` is 3.2 GiB
(3 450 062 241 bytes), far beyond GitHub's 100 MB file limit. It is
published by Meta on Hugging Face, gated: request access on
<https://huggingface.co/facebook/sam3> with a Hugging Face account, then
make that account's access token available, either as `HF_TOKEN` or
saved by `hf auth login` in `~/.cache/huggingface/token`.

The module downloads `sam3.pt` on first use into the central folder
`$SAM_MODELS_DIR/sam3` (by default `~/models/sam/sam3`), or into
**weights_dir**; `-n` disables this. It can also be fetched beforehand:

```sh
mkdir -p ~/models/sam/sam3
cd ~/models/sam/sam3
curl -L -O -H "Authorization: Bearer $HF_TOKEN" \
    https://huggingface.co/facebook/sam3/resolve/main/sam3.pt
```

```sh
# The central folder, downloading sam3.pt if missing:
i.sam3.opencl group=rgb output=segments

# One checkpoint file:
i.sam3.opencl group=rgb checkpoint=~/models/sam/sam3/sam3.pt output=segments
```

## Building

```sh
make MODULE_TOPDIR=$HOME/dev/grass
```

or install into the user's addon directory with
`g.extension extension=i.sam3.opencl url=/path/to/i.sam3.opencl`.
Requires an OpenCL 1.2 platform and its development headers (e.g.
`ocl-icd-opencl-dev`), libcurl (`libcurl4-openssl-dev`) and a GRASS GIS
build with OpenCL support. With Mesa on AMD GPUs, export
`RUSTICL_ENABLE=radeonsi` (or the matching driver) to expose the GPU
through rusticl.

## Tests

```sh
I_SAM3_OPENCL_CHECKPOINT=~/models/sam/sam3/sam3.pt pytest tests
```

Without `I_SAM3_OPENCL_CHECKPOINT`, only the input validation tests run.

`tests/reference_compare.py` compares the module's debug dumps (written
when `I_SAM3_OPENCL_DUMP` names a directory) with the PyTorch `sam3`
package; it needs `torch`, `torchvision` and `sam3`, which the module
itself does not.
