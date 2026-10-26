# TODO

## Device adaptation: remaining work

`profile=` and `precision=` were added on 2026-10-26 (see the manual,
"Device profile and precision"). Every GEMM variant (generic, large,
CPU, float and half weights) matches a host reference on PoCL, rusticl
and Intel compute-runtime. Still open:

- [ ] Full-model check of `profile=cpu` (speed against the GPU kernels on
      PoCL) and of `precision=fp16` (masks against fp32) on a real crop;
      the checkpoint-gated tests cover the rectangles image.
- [ ] Install into `/usr/local/grass86` once that check passes (only the
      dev tree is built).
- [ ] Intel GPUs: `gemm_big` spills registers at SIMD8 on Gen12
      (8x8 block, 128 registers); add a smaller register block for the
      `igpu` profile on Intel.
- [ ] Intel XMX through `cl_intel_subgroup_matrix_multiply_accumulate`
      (Xe2/Xe3, Arc Pro B50/B60, Panther Lake).
- [ ] Host BLAS (`cblas_sgemm` on mapped buffers) as a further `cpu` path.
- [ ] Kernel fusion: layernorm with the residual add; softmax inside the
      attention product (flash-attention style).
- [ ] `egpu`: overlap transfers with compute; untested on a real eGPU.
- [ ] FP16 activations (not only weights) where `cl_khr_fp16` exists.
