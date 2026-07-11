# GFX1151 Anima performance report

## Result

The production path remains BF16 dense Anima weights, Flash Attention, full GPU residency, `torch.compile`, and the Spectrum `simple` sampler workflow. At 1024x1024, 30 configured steps and CFG 5, the measured steady-state prompt time was 45.49 seconds for the ordinary KSampler and 27.80 seconds for the saved Spectrum workflow. Spectrum executed 17 model forwards while retaining the configured 30-step schedule, a 38.9% end-to-end reduction.

The saved workflow is `user/default/workflows/Anima_INT8_ConvRot_Spectrum_simple_30step.json`.

## Attention research

### Current baseline

The Anima 2B denoiser uses 28 blocks, 16 heads and head dimension 128. At 1024x1024 the main self-attention sequence is 4096 tokens; cross-attention uses 4096 queries and 512 text keys. At 1536x1536 self-attention grows to 9216 tokens.

The local PyTorch 2.14 ROCm 7.15 build and standalone Flash Attention already provide a strong gfx1151 baseline:

| BF16 shape, B=2 H=16 D=128 | Flash | New exact Triton | Decision |
|---|---:|---:|---|
| self 4096x4096 | 9.376 ms | 18.138 ms | keep Flash |
| cross 4096x512 | 1.763 ms | 2.445 ms | keep Flash |
| self 9216x9216 | 48.137 ms | 90.257 ms | keep Flash |

The exact Triton implementation is useful as a wave64-compatible correctness baseline, but automatic dispatch would reduce performance.

### 2026 methods

[PISA](https://arxiv.org/abs/2602.01077) is the best match for an existing Anima checkpoint under the constraints of no step reduction, no training and no temporal feature-cache bypass. It preserves the full attention span by evaluating critical blocks exactly and approximating the tail inside the same softmax numerator and denominator. The paper reports 1.2x end-to-end acceleration on FLUX. The official implementation is primarily Hopper-optimized, and its small gfx1151 probe did not finish compiling within one minute.

A gfx1151 prototype was implemented with block size 64, exact online-softmax contributions, block-centred zeroth-order tail terms and the global first-order correction. Its Triton statistics stage takes roughly 0.5 ms at T=4096, but the mixed exact/approximate stage is not fused and takes seconds. It is therefore available only as a benchmark, never as an automatic model patch. Random-QKV cosine similarity at 15.625% exact blocks was about 0.81, so real Anima quality validation is mandatory even after fusion.

[ElasticDiT SSBA](https://arxiv.org/abs/2605.15684) hard-drops block interactions and was evaluated as part of a model trained around elastic depth, compression and sparse attention. It is not a quality-preserving drop-in replacement for dense Anima.

[DiffSparse](https://arxiv.org/abs/2604.03674) learns layer/timestep sparsity and reuses token features. It requires Anima-specific training and violates the no-feature-cache constraint.

[SAFE-DiT](https://arxiv.org/abs/2606.29360) removes redundant attention masks exactly, but Anima main self-attention already uses `mask=None`. Its remaining speedup uses selective state reuse and is not applicable under the current constraint.

[TurboDiffusion](https://arxiv.org/abs/2512.16093) combines step distillation, trainable sparse-linear attention, W8A8 and SageAttention. Its headline speedup cannot be transferred to the existing checkpoint without changing the step/model constraints.

## Rejected experiments

| Experiment | Measurement | Decision |
|---|---|---|
| SpargeAttention gfx1151, T=4096, top-k 0.1 | 11.825 ms vs Flash 8.993 ms; random-QKV cosine 0.337 | reject |
| Exact cross-attention K/V projection reuse | 45.83 s median vs 45.49 s baseline | reject and remove |
| Exact gfx1151 Triton full attention | 1.39x to 1.94x slower than Flash | benchmark only |
| Staged PISA | preparation fast, mixed stage thousands of times slower than Flash | research only |
| Extending INT8 to the second MLP | 14.6 ms vs BF16 11.2 ms | keep BF16 dense |

The K/V experiment was exact and execution-scoped: 28 layer-specific projected K/V pairs were created at sampling entry and passed into the compiled model, then released in `finally`. The extra graph inputs and preparation cost outweighed 56 avoided projection GEMMs per step on the 2B checkpoint, so all implementation seams were removed after measurement.

## CPU/GPU mixed-memory policy

gfx1151 is an APU with shared physical DRAM, but PyTorch CPU and `cuda` tensors still have distinct storage and synchronization semantics. CPU offload causes copies, pin registration, synchronization and contention for the same DRAM bandwidth. During a GPU-saturated generation, even the local HTTP server and PowerShell became temporarily unresponsive, confirming that concurrent CPU tensor work is not free.

For the measured approximately 4.9 GB working set, use this partition:

- CPU: tokenization, prompt graph, checkpoint/file cache and lightweight orchestration.
- GPU: Qwen text encoder, Anima LLM adapter, complete denoiser, latent/conditioning tensors and Wan VAE.
- Keep the denoiser resident for all 30 steps. Do not stage layers through pinned host buffers when it fits.
- Use async offload only for models that cannot remain resident; on an APU, transfer and GEMM streams compete for shared DRAM.

ROCm reports about 102 GB GPU memory on a machine with about 65 GB physical RAM. Automatic DynamicVRAM admission should eventually clamp its effective budget to the smaller of HIP free memory and available physical memory minus a safety reserve. Explicit `--gpu-only` should continue to mean full residency and should not be overridden. This is an upstream-worthy memory-correctness change, but it does not improve the current explicit GPU-only Anima path and was not mixed into the performance branch without a dedicated default-mode benchmark.

## Timing interpretation

The progress bar is not a valid end-to-end GPU timer on this asynchronous ROCm path. A steady-state run displayed 30 steps in 2.81 seconds while the prompt took 45.25 seconds. Work queued during sampling completes at later synchronization points, so optimization acceptance uses prompt completion time and fresh-process warmups, not tqdm iteration rate.
