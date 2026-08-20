# RDNA35 PISA KSampler implementation plan

## Corrected premise

The earlier `69.12 s -> 68.63 s` workflow comparison did not measure PISA. The
Anima call flattened `(T, H, W)` before attention, so the plugin received
`attention_token_shape=(9216,)` and all eligible self-attention calls fell back
to Flash Attention. That result must not be attributed to PISA.

The direct attention benchmark remains useful only as a kernel microbenchmark:

| Backend | BF16 `B=2,H=16,T=9216,D=128` |
|---|---:|
| Flash Attention | 46.411 ms |
| PISA 23/144 | 37.917 ms |

## Evidence from the corrected local path

The local Cosmos/Anima path now preserves the original token shape before
flattening. Warm 1536x1536 Spectrum measurements in one resident process were:

| Checkpoint/path | 1 step | 30 steps |
|---|---:|---:|
| INT8 ConvRot mixed, Flash | 6.326 s | 74.491 s |
| INT8 ConvRot mixed, PISA | 6.452 s | 74.354 s |
| BF16, Flash | 6.199 s | 72.017 s |
| BF16, PISA | - | 70.841 s |
| INT8 checkpoint load-time dense, PISA | 6.152 s | 74.853 s |

The 30-step values show that the current hybrid PISA path saves only 1.6% on
the BF16 checkpoint and is inside run variance on the INT8 ConvRot path. PISA
therefore cannot yet be claimed to provide a 10% KSampler improvement.

Rejected experiments:

- `torch.compile(fullgraph=True)`: 6.410 s versus 6.152 s for the best default
  one-step path.
- `torch.compile(mode="reduce-overhead")`: fatal process failure on Windows
  ROCm; do not enable CUDA graph mode in the launcher.
- gfx1151 radix-4 Triton FHT256: 4.838 ms versus 1.089 ms for the existing
  dense Hadamard rotation at `(M,K)=(18432,2048)`; the patch was removed.
- Spectrum SEA `refresh_ratio=0.45`: 78.567 s versus 72.202 s for the normal
  window schedule with the same seed; the setting was not retained.

## Phase 0: benchmark validity and call accounting

No further kernel result is accepted without runtime proof that PISA executed.

Add to `ComfyUI-RDNA35-Attention`:

```text
rdna35_block_attention/pisa_runtime.py
rdna35_block_attention/sampler_profiler.py
scripts/bench_anima_ksampler.py
tests/test_profiler_accounting.py
```

Record per sample:

```text
actual model forwards
PISA calls by transformer layer
Flash calls by transformer layer
cross-attention calls
fallback counts grouped by reason
B, H, T, D, dtype and stride
whole PISA-call GPU events
```

For the current workflow the validity gate is:

```text
expected PISA calls = actual forwards * 24
cross-attention PISA calls = 0
layers 0..3 PISA calls = 0
eligible fallbacks = 0
```

If `pisa_calls == 0` or the histogram differs from the gate, the benchmark must
return `INVALID BENCHMARK` and must not report a speedup.

## Phase 1: direct model-local Anima integration

The production plugin must not depend on optional metadata surviving the
generic `optimized_attention_override` path. Clone the `ModelPatcher`, locate
the validated 28-block Anima diffusion model, and install object patches only
on `blocks[4:28].self_attn.attn_op`.

The patched op knows the block index and self-attention ownership at patch time.
It validates BF16, gfx1151, `T=9216`, 16 heads, head dimension 128, mask-free
forward-only execution, then calls `rdna35_pisa_ck.forward_spatial_bhtd`
directly. An eligible call that cannot execute PISA is an error, not a silent
Flash fallback. Other models, resolutions, dtypes and cross-attention retain
the previous backend.

Keep the small Cosmos token-shape propagation change as an upstream-facing
attention-context improvement, but do not make the plugin's production
correctness depend on it.

## Phase 2: runtime report and reproducible harness

Attach per-clone runtime state with a bounded lifetime. Expose a post-sampling
report node or execution hook that distinguishes:

```text
armed -> executed -> verified
inactive
failed
```

The benchmark harness must submit API-format prompts, change only the seed to
avoid node caching, separate cold compile from warm runs, and execute an ABBA
order. Save environment, commits, wheel build information, workflow hash,
actual-forward count, call accounting, GPU time, sampler time and prompt time.

## Phase 3: profile the complete hybrid PISA call

Use asynchronous HIP events and synchronize once at sampler completion. Measure:

```text
spatial QKV pack
block statistics
centered H_sum
routing scores and top-k
mask construction
approximate FlexAttention
exact FlexAttention
LSE merge and correction
output unpack
```

Only optimize components whose measured contribution can move the 30-step
sampler by at least 1%.

## Phase 4: hybrid cleanup

In measured order:

1. Fuse spatial pack with Q/K/V block statistics.
2. Remove the full-size centered-K tensor and accumulate `H_sum` directly in
   FP32.
3. Replace Python/PyTorch routing, top-k, sorting and bool-mask construction
   with compact native metadata for the fixed 23/144 profile.
4. Fuse final correction, BF16 conversion and raster-order output.

Do not add persistent tensor caches. Workspaces must be execution-scoped and
must not change CFG batching or offload behavior.

## Phase 5: persistent block-major Anima layout

If pack/unpack remains material after Phase 3, convert the residual stream once
before layer 4, keep layers 4..27 in the same 8x8 block-major token order, and
convert back once before the final layer. RoPE and positional data must use the
same permutation. Validate first with dense 144-block attention before enabling
the approximate 23-block profile.

## Phase 6: fused gfx1151 PISA kernel

After call accounting and hybrid profiling are complete, replace the two
FlexAttention calls and Python merge with a native gfx1151 path:

```text
statistics -> compact top-23 routing -> fused exact/approximate online softmax
           -> first-order correction -> BF16 output
```

The 23 exact blocks are compile-time fixed. Keep deterministic tie-breaking,
FP32 accumulation, the current CK/Flex implementation as a safe fallback, and
the `torch.library` fake implementation required by outer `torch.compile`.

## Origami workstream

Runtime Origami work is separated from Stream-K. The installed hipBLASLt has
not selected an explicitly tagged Stream-K solution, and method 2 previously
regressed workflow time. Keep `TENSILE_SOLUTION_SELECTION_METHOD=0` until a
real Stream-K solution is logged and wins repeated measurements.

Next, build a product-shape oracle from generated Anima GEMMs and compare
Origami top-k 4/6/8/12 with `EXHAUSTIVE` in separate processes and clean cache
directories. Report shape-frequency-weighted regret, top-k oracle recall, cold
compile time, and five-run warm median/CV. Record both the 20 WGP-like units
reported by PyTorch and the 40 physical CUs modeled by Origami.

## Acceptance gates

Integration:

- One actual Anima forward produces exactly 24 PISA calls.
- A 17-forward Spectrum sample produces exactly 408 PISA calls.
- Eligible fallback, cross-attention hit, and layer 0..3 hit counts are zero.
- All intermediate and final outputs are finite.

Performance:

- Warm ABBA median over at least five samples.
- At least 10% KSampler reduction on gfx1151 for the specified workflow.
- p90 does not regress and compile time is reported separately.
- A change is retained only if its isolated contribution is outside run noise.

Quality:

- Same-seed image matrix, not one image.
- No regression below the current validated 23/144 profile's p05 SSIM, RGB
  cosine, LPIPS and detail-retention distribution.
- Unknown shapes, dtypes, model layouts and wheel ABIs fail closed.

## PR order

1. Invalidate the earlier fallback-derived workflow claim in documentation.
2. Add direct model-local Anima PISA integration.
3. Add runtime accounting and invalid-benchmark enforcement.
4. Publish the first verified KSampler baseline.
5. Optimize measured hybrid overhead.
6. Add persistent block-major layout if justified.
7. Add fused native gfx1151 PISA and performance CI.

