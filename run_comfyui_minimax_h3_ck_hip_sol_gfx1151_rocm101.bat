@echo off
setlocal
cd /d C:\ComfyUI

set "PYTHONNOUSERSITE=1"
set "PYTHON_EXE=C:\ComfyUI\.venv-rocm101\Scripts\python.exe"
set "PYTHON_SITE_PACKAGES=C:\ComfyUI\.venv-rocm101\Lib\site-packages"
set "COMFY_KITCHEN_DEV=C:\Dev\comfy-kitchen-pr94-pr115-pr116-sol-hip"
set "PYTHONPATH=%COMFY_KITCHEN_DEV%;%PYTHONPATH%"
set "ROCM_CORE=%PYTHON_SITE_PACKAGES%\_rocm_sdk_core"
set "ROCM_DEVEL=%PYTHON_SITE_PACKAGES%\_rocm_sdk_devel"
set "ROCM_LIBS=%PYTHON_SITE_PACKAGES%\_rocm_sdk_libraries"

set "PATH=%ROCM_LIBS%\bin;%ROCM_CORE%\bin;%ROCM_DEVEL%\bin;%PATH%"
set "INCLUDE=%ROCM_CORE%\include;%ROCM_DEVEL%\include;%INCLUDE%"
set "LIB=%ROCM_CORE%\lib;%ROCM_DEVEL%\lib;%ROCM_LIBS%\lib;%LIB%"
set "ROCM_HOME=%ROCM_CORE%"
set "ROCM_PATH=%ROCM_CORE%"
set "HIP_PATH=%ROCM_CORE%"
set "HIP_DEVICE_LIB_PATH=%ROCM_CORE%\lib\llvm\amdgcn\bitcode"

set "PYTORCH_ROCM_ARCH=gfx1151"
set "GPU_ARCHS=gfx1151"
set "GPU_TARGETS=gfx1151"
set "CMAKE_HIP_ARCHITECTURES=gfx1151"
set "HIP_ARCHITECTURE=gfx1151"
set "MAX_JOBS=32"
set "CMAKE_BUILD_PARALLEL_LEVEL=32"

set "TORCH_BLAS_PREFER_HIPBLASLT=1"
set "TORCH_BLAS_PREFER_CUBLASLT=1"
set "ROCBLAS_USE_HIPBLASLT=1"
set "CUBLASLT_WORKSPACE_SIZE=262144"
set "HIPBLASLT_TUNING_USER_MAX_WORKSPACE=268435456"
set "COMFYUI_ENABLE_PYTORCH_VAE_ON_AMD=1"
set "COMFYUI_ENABLE_MIOPEN=1"
set "PYTHONFAULTHANDLER=1"
set "TQDM_MININTERVAL=1"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512"
if exist "%ROCM_LIBS%\bin\hipdnn_plugins\engines" set "HIPDNN_PLUGIN_DIR=%ROCM_LIBS%\bin\hipdnn_plugins\engines"
if exist "%ROCM_LIBS%\bin\hipdnn_plugins\heuristics" set "HIPDNN_HEURISTIC_PLUGIN_DIR=%ROCM_LIBS%\bin\hipdnn_plugins\heuristics"

set "AITER_ENABLE_HIP=1"
set "AITER_TRITON_ONLY=0"
set "AITER_USE_SYSTEM_TRITON=1"
set "FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE"
set "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0"
set "TORCH_ROCM_FA_PREFER_CK=0"
set "SOL_ATTN_DISABLE_AOTRITON_SDPA=1"

rem Ryzen AI Max+ 395 / 128GB UMA: pin budget is 40%% of RAM (~51GB) and shares
rem the same DRAM as HIP. Stream H3 weights from NVMe instead of locking them.
rem cache-ram values are free-memory headroom, not cache size.
if not defined COMFY_H3_RESERVE_VRAM set "COMFY_H3_RESERVE_VRAM=8"
if not defined COMFY_H3_DYNAMIC_HEADROOM set "COMFY_H3_DYNAMIC_HEADROOM=4"
if not defined COMFY_H3_CACHE_ACTIVE set "COMFY_H3_CACHE_ACTIVE=16"
if not defined COMFY_H3_CACHE_INACTIVE set "COMFY_H3_CACHE_INACTIVE=32"
if not defined COMFY_H3_PORT set "COMFY_H3_PORT=8193"

echo [H3 UMA] Validating the dedicated Comfy-Kitchen checkout...
"%PYTHON_EXE%" -c "import pathlib, torch, comfy_kitchen as ck; root=pathlib.Path(r'%COMFY_KITCHEN_DEV%').resolve(); loaded=pathlib.Path(ck.__file__).resolve(); hip=ck.list_backends().get('hip', {}); assert loaded.is_relative_to(root), f'wrong Comfy-Kitchen: {loaded}'; assert hip.get('available') and 'sol_attn' in hip.get('capabilities', []), hip; q=torch.randn(1,256,4,128,device='cuda',dtype=torch.bfloat16); out=ck.sol_attn(q,q,q,tau=1.3); torch.cuda.synchronize(); assert torch.isfinite(out).all() and torch.count_nonzero(out); print(f'[H3 UMA] checkout={loaded} arch={torch.cuda.get_device_properties(0).gcnArchName} output={tuple(out.shape)}')"
if errorlevel 1 (
    echo [H3 UMA] Preflight failed. ComfyUI was not started.
    exit /b 1
)

echo [H3 UMA] Port=%COMFY_H3_PORT% FastDisk=ON NoPin=ON AOTriton-SDPA=DISABLED CK-attention=ON DynamicVRAM=ON
"%PYTHON_EXE%" -u main.py --listen 0.0.0.0 --port %COMFY_H3_PORT% --enable-dynamic-vram --fast-disk --disable-pinned-memory --disable-nvml-pressure --vram-headroom %COMFY_H3_DYNAMIC_HEADROOM% --reserve-vram %COMFY_H3_RESERVE_VRAM% --disable-async-offload --cache-ram %COMFY_H3_CACHE_ACTIVE% %COMFY_H3_CACHE_INACTIVE% --use-ck-attention --disable-xformers %*

endlocal
