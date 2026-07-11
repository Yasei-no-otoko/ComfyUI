# ComfyUI RDNA35 Attention Research

ComfyUI custom nodes for fixed 64-token block-diagonal self-attention on PyTorch ROCm with an optional Triton forward kernel. This is not a port of NVIDIA Blackwell TLX code.

The package also contains two isolated gfx1151 research paths. Neither replaces normal ComfyUI attention automatically:

- Exact full attention with an online-softmax Triton kernel for `[BH,Q,D] x [BH,K,D]`.
- A training-free PISA prototype based on the 2026 exact-or-approximate attention method.

## What This Implements

The operation splits the sequence into fixed blocks of 64 tokens. Tokens in block `i` attend only to keys and values from block `i`. This is exact for fixed block-diagonal attention, but it is not equivalent to normal full attention because cross-block attention is removed.

The Triton path is forward/inference only. There is no custom autograd or backward path.

## Nodes

- `RDNA35 Block Attention Diagnostics`: reports PyTorch, HIP, device, best-effort gfx target, Triton availability, and RDNA3.5 detection.
- `RDNA35 Patch Model Attention`: installs a model-local `optimized_attention_override` on a cloned MODEL. It never globally monkey-patches ComfyUI attention.
- `RDNA35 Fixed Block Attention Benchmark`: creates synthetic Q/K/V tensors, compares reference, dispatch, PyTorch SDPA with a block-diagonal mask, and normal PyTorch full SDPA. Full SDPA is reported as a semantic contrast, not as an exact replacement.
- `RDNA35 Exact Full Attention Benchmark`: compares the gfx1151 online-softmax kernel with PyTorch SDPA for Anima-like self- and cross-attention shapes.
- `RDNA35 PISA Attention Prototype Benchmark`: reports latency and numerical deviation for the staged PISA implementation. It is intentionally not exposed as a model patch until the mixed exact/approximate loop is fused and faster than Flash Attention.

## Measured gfx1151 results

On the local PyTorch 2.14 ROCm 7.15 stack with BF16 and `B=2,H=16,D=128`:

| Shape | Flash Attention | Exact Triton | Ratio |
|---|---:|---:|---:|
| Q=4096, K=4096 | 9.376 ms | 18.138 ms | 1.935x slower |
| Q=4096, K=512 | 1.763 ms | 2.445 ms | 1.387x slower |
| Q=9216, K=9216 | 48.137 ms | 90.257 ms | 1.875x slower |

The exact kernel is retained as a portable correctness baseline but is not selected for generation. The staged PISA preparation kernel takes about 0.5 ms at `T=4096`, while the current Python/PyTorch mixed-attention stage takes seconds. A fused wave64 implementation is still required before PISA can be considered for model dispatch.

PISA remains approximate. At 15.625% exact blocks on random `T=4096,D=128` tensors, cosine similarity against dense attention was about 0.81. Production quality must be measured on real Anima Q/K/V and generated images even after the kernel becomes fast.

## Install

Place this folder under:

```powershell
C:\ComfyUI\custom_nodes\ComfyUI-RDNA35-FixedBlockAttention
```

Restart ComfyUI. The nodes appear under `RDNA35/Fixed Block Attention`.

Recommended runtime:

- PyTorch ROCm build
- Triton compatible with that PyTorch ROCm build
- AMD RDNA3.5 target such as `gfx1150`, `gfx1151`, or `gfx1152`

PyTorch ROCm still uses `torch.cuda` APIs and `device="cuda"` strings. ROCm detection is based on `torch.version.hip`.

## Patch Safety

`RDNA35 Patch Model Attention` defaults to `exact_only`. In that mode, ordinary ComfyUI attention calls are left unchanged unless a call explicitly declares:

```python
rdna35_attention_semantics = "fixed_block_diagonal"
```

`experimental_force_block_local` is opt-in. It still refuses calls that are not explicitly marked or otherwise proven self-attention. Cross-attention is never intentionally converted to block-local attention.

If a safe local model patch cannot be installed, the node returns the original model and includes the reason in the `info` output.

## Optimized Dispatch Conditions

The Triton kernel is used only when all of these are true:

- PyTorch ROCm/HIP is detected through `torch.version.hip`
- Q/K/V are on the same `cuda` device, which is the PyTorch ROCm device type
- Triton imports successfully
- dtype is `float16` or `bfloat16`
- head dimension is 32, 64, or 128
- `block_size == 64`
- Q/K/V shapes match and represent self-attention
- layout is supported and normalized to contiguous `[BH,T,D]`
- no arbitrary mask is passed
- no input has `requires_grad=True`

Otherwise dispatch falls back to the PyTorch reference implementation with a reason. The ComfyUI patch falls back to the original attention backend for calls that are not known to be fixed block-diagonal.

## Limitations

- Forward/inference only
- Fixed `block_size=64`
- No arbitrary mask in the Triton path
- No cross-attention conversion
- No CUDA extensions
- No NVIDIA Blackwell TLX features such as async dot, TMA, TMEM, mBarrier, or tcgen05
- No silent replacement of normal full attention

## Tests

The runtime Python on this machine may not have `pytest`; the tests are `unittest` compatible:

```powershell
$py = 'C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\python.exe'
& $py -m unittest discover -s C:\ComfyUI\custom_nodes\ComfyUI-RDNA35-FixedBlockAttention\tests -v
```

Reference-only tests run on CPU. Triton tests skip automatically when ROCm/Triton is unavailable.

## Benchmark

```powershell
$py = 'C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\python.exe'
& $py C:\ComfyUI\custom_nodes\ComfyUI-RDNA35-FixedBlockAttention\scripts\bench_fixed_block_attention.py --tokens 256 --head-dim 64 --dtype float16 --mode auto
```

The benchmark reports:

- PyTorch reference loop latency for fixed block-diagonal attention
- PyTorch SDPA with a block-diagonal mask, which has the same fixed-block semantics
- PyTorch full SDPA, which is normal full attention and not semantically equivalent
- Triton latency and speedups only if the Triton backend actually runs
- The full-SDPA semantic delta vs fixed-block reference, so the output difference is visible

## Primary References

- [PyTorch TLX Block Attention blog](https://pytorch.org/blog/tlx-block-attention-a-warp-specialized-blackwell-kernel-for-fixed-block-sparse-self-attention/)
- [PyTorch HIP semantics](https://docs.pytorch.org/docs/2.12/notes/hip.html)
- [ROCm Triton install docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-triton.html)
- [ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- [ROCm RDNA3.5 optimization docs](https://rocm.docs.amd.com/en/7.13.0-preview/reference/system-optimization/rdna3-5.html)
- [ComfyUI custom node backend docs](https://docs.comfy.org/custom-nodes/backend/server_overview)
- [PISA paper (arXiv:2602.01077)](https://arxiv.org/abs/2602.01077)
- [Official PISA implementation](https://github.com/xie-lab-ml/piecewise-sparse-attention)
