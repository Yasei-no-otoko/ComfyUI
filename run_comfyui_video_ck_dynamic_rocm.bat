@echo off
setlocal
cd /d C:\ComfyUI

set "PYTHON_EXE=C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\python.exe"
set "PYTHON_SITE_PACKAGES=C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\Lib\site-packages"
set "PYTHONPATH=C:\ComfyUI\custom_nodes\comfy-kitchen.disabled;%PYTHONPATH%"
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
set "MAX_JOBS=32"
set "CMAKE_BUILD_PARALLEL_LEVEL=32"

set "TORCH_BLAS_PREFER_HIPBLASLT=1"
set "TORCH_BLAS_PREFER_CUBLASLT=1"
set "ROCBLAS_USE_HIPBLASLT=1"
set "CUBLASLT_WORKSPACE_SIZE=262144"
set "HIPBLASLT_TUNING_USER_MAX_WORKSPACE=268435456"
set "COMFYUI_ENABLE_PYTORCH_VAE_ON_AMD=1"
set "COMFYUI_ENABLE_MIOPEN=1"
set "TORCHINDUCTOR_COMPILE_THREADS=1"
set "PYTHONFAULTHANDLER=1"
set "TQDM_MININTERVAL=1"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512"
if exist "%ROCM_LIBS%\bin\hipdnn_plugins\engines" set "HIPDNN_PLUGIN_DIR=%ROCM_LIBS%\bin\hipdnn_plugins\engines"
if exist "%ROCM_LIBS%\bin\hipdnn_plugins\heuristics" set "HIPDNN_HEURISTIC_PLUGIN_DIR=%ROCM_LIBS%\bin\hipdnn_plugins\heuristics"

set "AITER_ENABLE_HIP=1"
set "AITER_TRITON_ONLY=0"
set "AITER_USE_SYSTEM_TRITON=1"

rem Keep external FlashAttention and AOTriton paths out of this comparison launcher.
set "FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE"
set "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0"
set "TORCH_ROCM_FA_PREFER_CK=0"

rem Keep enough headroom for H3 VideoVAE decode while Dynamic VRAM streams model weights.
if not defined COMFY_VIDEO_RESERVE_VRAM set "COMFY_VIDEO_RESERVE_VRAM=8"
if not defined COMFY_VIDEO_DYNAMIC_HEADROOM set "COMFY_VIDEO_DYNAMIC_HEADROOM=4"

echo [H3] AITER=ON Triton=ON ComfyKitchen-INT8=ON DynamicVRAM=ON
"%PYTHON_EXE%" -u main.py --enable-manager --listen 0.0.0.0 --enable-dynamic-vram --vram-headroom %COMFY_VIDEO_DYNAMIC_HEADROOM% --reserve-vram %COMFY_VIDEO_RESERVE_VRAM% --disable-async-offload --cache-ram --use-ck-attention %*
