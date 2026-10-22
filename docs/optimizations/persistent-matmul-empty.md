# Persistent MatMul: remove redundant zero initialization

## Mechanism

The batch-invariant Triton `linear_persistent` kernel fully writes every valid
output element. OPset replaces its output allocation from `torch.zeros` to
`torch.empty`, avoiding a separate zero-fill kernel without changing the
ordinary ACLNN MatMul path.

## Canonical results

[Original measured report and raw result links](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/vllm-hust-opset/snapshot/docs/optimizations/persistent-matmul-empty.md).
The complete frozen report retains run identities, correctness checks, cold-start
outliers and negative controls. No measurement is duplicated in this MOD.

## Boundary

This optimization applies only when the verified persistent Triton kernel is
selected. Unknown source structures are rejected, and no claim is made for
ordinary ACLNN MatMul or unrelated linear implementations.
