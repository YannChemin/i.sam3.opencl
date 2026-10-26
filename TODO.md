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
- [ ] Intel GPUs: on mains power `gemm_big` reaches about 970 GFLOP/s
      (46 % of peak) on the Iris Xe; a 4x4-block variant reached 720 and
      was dropped on 2026-10-26. Next: `cl_intel_subgroups` block reads,
      and XMX on Xe2 and later. (Timings on battery are about 14x lower:
      benchmark on mains power only.)
- [ ] Intel XMX through `cl_intel_subgroup_matrix_multiply_accumulate`
      (Xe2/Xe3, Arc Pro B50/B60, Panther Lake).
- [ ] Host BLAS (`cblas_sgemm` on mapped buffers) as a further `cpu` path.
- [ ] Kernel fusion: layernorm with the residual add; softmax inside the
      attention product (flash-attention style).
- [ ] `egpu`: overlap transfers with compute; untested on a real eGPU.
- [ ] FP16 activations (not only weights) where `cl_khr_fp16` exists.
