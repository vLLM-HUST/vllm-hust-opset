# Linear + SwiGLU Graph fusion

## Mechanism

The baseline materializes the `[M, 2N]` gate/up result from an unquantized
GEMM and then launches `npu_swiglu`. The candidate keeps gate and up
accumulators in one persistent Triton kernel, applies
`gate * sigmoid(gate) * up`, and stores only `[M, N]`.

The validated path accepts `M <= 128`. Larger dynamic prefill shapes and
unknown graph structures fail closed to native GEMM plus `npu_swiglu`.

## Canonical results

[Original measured report and raw result links](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/vllm-hust-opset/snapshot/docs/optimizations/linear-swiglu-graph.md).
The complete frozen report retains run identities, correctness checks, cold-start
outliers and negative controls. No measurement is duplicated in this MOD.

## Boundary and negative evidence

- These results cover unquantized Graph/Triton execution, not ACLNN or
  quantized linear paths.
- A no-workload-warmup experiment exposed one-time Triton JIT cost and yielded
  negative candidate TPOT; cold start is deliberately separated from the
  steady-state claim.
- Large-M fusion caused CBUF overflow and, for some multi-tile shapes,
  incorrect values. The implementation therefore falls back for `M > 128`.
