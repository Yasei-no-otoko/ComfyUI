@echo off
setlocal
cd /d C:\ComfyUI

set "PYTHON_EXE=C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\python.exe"
set "PYTHON_SITE_PACKAGES=C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\Lib\site-packages"
set "AITER_ROOT=C:\aiter"
set "PYTHONPATH=%AITER_ROOT%;C:\ComfyUI\custom_nodes\comfy-kitchen.disabled;%PYTHONPATH%"
set "ROCM_CORE=%PYTHON_SITE_PACKAGES%\_rocm_sdk_core"
set "ROCM_DEVEL=%PYTHON_SITE_PACKAGES%\_rocm_sdk_devel"
set "ROCM_LIBS=%PYTHON_SITE_PACKAGES%\_rocm_sdk_libraries_gfx1151"

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

rem Use the Windows HIP/CK path from the patched C:\aiter checkout.
set "AITER_ENABLE_HIP=1"
set "AITER_TRITON_ONLY=0"
set "AITER_USE_SYSTEM_TRITON=1"
set "ENABLE_CK=1"
set "AITER_JIT_DIR=C:\ComfyUI\user\.cache\aiter_jit_gfx1151"
set "MAX_JOBS=32"
set "CMAKE_BUILD_PARALLEL_LEVEL=32"

rem Let Origami analytically rank the Triton GEMM candidates before max-autotune.
set "TORCHINDUCTOR_ORIGAMI=1"
set "TORCHINDUCTOR_ORIGAMI_TOPK=8"
set "TORCHINDUCTOR_MAX_AUTOTUNE=1"
set "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1"
set "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=DEFAULT"
rem Coordinate descent also tunes non-GEMM Triton kernels and can fail to converge.
set "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0"
rem PyTorch 2.14 Windows ROCm workers can terminate during full-model lowering.
rem Keep Inductor compilation in-process; native builds still use 32 workers above.
set "TORCHINDUCTOR_COMPILE_THREADS=1"
set "TORCHINDUCTOR_CACHE_DIR=C:\ComfyUI\user\.cache\torchinductor_origami_ck_aiter_gfx1151"

set "TORCH_BLAS_PREFER_HIPBLASLT=1"
set "TORCH_BLAS_PREFER_CUBLASLT=1"
set "ROCBLAS_USE_HIPBLASLT=1"
set "CUBLASLT_WORKSPACE_SIZE=262144"
set "HIPBLASLT_TUNING_USER_MAX_WORKSPACE=268435456"
set "COMFYUI_ENABLE_PYTORCH_VAE_ON_AMD=1"
set "COMFYUI_ENABLE_MIOPEN=1"
set "PYTHONFAULTHANDLER=1"
set "TQDM_MININTERVAL=1"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8"
if exist "%ROCM_LIBS%\bin\hipdnn_plugins\engines" set "HIPDNN_PLUGIN_DIR=%ROCM_LIBS%\bin\hipdnn_plugins\engines"
if exist "%ROCM_LIBS%\bin\hipdnn_plugins\heuristics" set "HIPDNN_HEURISTIC_PLUGIN_DIR=%ROCM_LIBS%\bin\hipdnn_plugins\heuristics"

rem flash_attn_2_cuda is the installed CK FlashAttention extension.
rem Keep AITER's Triton FlashAttention disabled so it does not replace the CK path.
set "FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE"
set "TORCH_ROCM_FA_PREFER_CK=0"

"%PYTHON_EXE%" -u main.py --enable-manager --listen 0.0.0.0 --gpu-only --cache-ram --use-flash-attention --disable-xformers --enable-triton-backend --fast autotune %*
