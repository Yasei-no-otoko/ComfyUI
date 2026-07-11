@echo off
setlocal
cd /d C:\ComfyUI

set "PYTHON_EXE=C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\python.exe"
set "PYTHON_SITE_PACKAGES=C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\Lib\site-packages"
set "ROCM_CORE=%PYTHON_SITE_PACKAGES%\_rocm_sdk_core"
set "ROCM_DEVEL=%PYTHON_SITE_PACKAGES%\_rocm_sdk_devel"
set "ROCM_LIBS=%PYTHON_SITE_PACKAGES%\_rocm_sdk_libraries_gfx1151"
set "XFORMERS_ROCM_PATH=%PYTHON_SITE_PACKAGES%\xformers"
set "XFORMERS_FLASH_ATTENTION_PATH=%XFORMERS_ROCM_PATH%\_flash_attn"
set "XFORMERS_FLASH_ATTENTION_AMD_PATH=%XFORMERS_FLASH_ATTENTION_PATH%\flash_attn_triton_amd"

set "PATH=%ROCM_LIBS%\bin;%ROCM_CORE%\bin;%ROCM_DEVEL%\bin;%PATH%"
set "INCLUDE=%ROCM_CORE%\include;%ROCM_DEVEL%\include;%INCLUDE%"
set "LIB=%ROCM_CORE%\lib;%ROCM_DEVEL%\lib;%ROCM_LIBS%\lib;%LIB%"
set "ROCM_HOME=%ROCM_CORE%"
set "ROCM_PATH=%ROCM_CORE%"
set "HIP_PATH=%ROCM_CORE%"

set "TORCH_BLAS_PREFER_HIPBLASLT=1"
set "TORCH_BLAS_PREFER_CUBLASLT=1"
set "TORCH_ROCM_FA_PREFER_CK=0"
set "COMFYUI_ENABLE_PYTORCH_VAE_ON_AMD=1"
set "ROCBLAS_USE_HIPBLASLT=1"
set "CUBLASLT_WORKSPACE_SIZE=262144"
set "HIPBLASLT_TUNING_USER_MAX_WORKSPACE=268435456"
if exist "%ROCM_LIBS%\bin\hipdnn_plugins\engines" set "HIPDNN_PLUGIN_DIR=%ROCM_LIBS%\bin\hipdnn_plugins\engines"
if exist "%ROCM_LIBS%\bin\hipdnn_plugins\heuristics" set "HIPDNN_HEURISTIC_PLUGIN_DIR=%ROCM_LIBS%\bin\hipdnn_plugins\heuristics"

set "PYTORCH_ROCM_ARCH=gfx1151;gfx1201"
set "GPU_TARGETS=gfx1151;gfx1201"
set "CMAKE_HIP_ARCHITECTURES=gfx1151;gfx1201"
set "HIP_ARCHITECTURE=gfx1151"

set "XFORMERS_HIP_FLASH_MIN_SEQ_LEN=4096"
set "COMFYUI_XFORMERS_ROCM_OP=auto"
set "COMFYUI_XFORMERS_ROCM_FLASH_MIN_SEQ_LEN=4096"
set "COMFYUI_XFORMERS_ROCM_SPLITK_MIN_KV=256"
set "COMFYUI_XFORMERS_ROCM_SPLITK_MAX_Q=32"
set "COMFYUI_XFORMERS_ROCM_CK_SPLITK_MAX_Q=1"
set "COMFYUI_XFORMERS_ROCM_CK_SPLITK_MAX_M=1024"

rem Stability profile for LTX/LightTricks video VAE decode on Windows ROCm.
rem This intentionally avoids --gpu-only, --highvram, --disable-dynamic-vram,
rem --enable-triton-backend, and --fast because the crash happened in native
rem ROCm/Triton code during VideoVAE tiled 3D decode.
set "PYTHONFAULTHANDLER=1"
set "PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512"
set "TQDM_MININTERVAL=1"

if not defined COMFY_VIDEO_RESERVE_VRAM set "COMFY_VIDEO_RESERVE_VRAM=8"

set "COMFY_VIDEO_VRAM_ARGS="
set "COMFY_VIDEO_DYNAMIC_ARGS=--enable-dynamic-vram"
set "COMFY_VIDEO_VAE_ARGS="
set "COMFY_VIDEO_BACKEND_ARGS="
set "COMFY_VIDEO_OFFLOAD_ARGS=--disable-async-offload"

:parse_args
if "%~1"=="" goto run_comfyui
if /I "%~1"=="legacy-lowvram" (
    set "COMFY_VIDEO_VRAM_ARGS=--lowvram"
    set "COMFY_VIDEO_DYNAMIC_ARGS=--disable-dynamic-vram"
    shift
    goto parse_args
)
if /I "%~1"=="dynamic-vram" (
    set "COMFY_VIDEO_VRAM_ARGS="
    set "COMFY_VIDEO_DYNAMIC_ARGS=--enable-dynamic-vram"
    shift
    goto parse_args
)
if /I "%~1"=="triton-fast" (
    set "COMFY_VIDEO_DYNAMIC_ARGS=--enable-dynamic-vram"
    set "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
    set "COMFY_VIDEO_BACKEND_ARGS=--enable-triton-backend --fast fp16_accumulation"
    set "COMFY_VIDEO_OFFLOAD_ARGS=--async-offload 2"
    shift
    goto parse_args
)
if /I "%~1"=="async-offload" (
    set "COMFY_VIDEO_OFFLOAD_ARGS=--async-offload 2"
    shift
    goto parse_args
)
if /I "%~1"=="cpu-vae" (
    set "COMFY_VIDEO_VAE_ARGS=--cpu-vae"
    set "COMFY_VIDEO_DYNAMIC_ARGS=--disable-dynamic-vram"
    shift
    goto parse_args
)
if /I "%~1"=="fp16-vae" (
    set "COMFY_VIDEO_VAE_ARGS=--fp16-vae"
    shift
    goto parse_args
)
if /I "%~1"=="bf16-vae" (
    set "COMFY_VIDEO_VAE_ARGS=--bf16-vae"
    shift
    goto parse_args
)
if /I "%~1"=="fp32-vae" (
    set "COMFY_VIDEO_VAE_ARGS=--fp32-vae"
    shift
    goto parse_args
)
set "COMFY_VIDEO_EXTRA_ARGS=%COMFY_VIDEO_EXTRA_ARGS% %1"
shift
goto parse_args

:run_comfyui
"%PYTHON_EXE%" -u main.py --enable-manager --listen 0.0.0.0 %COMFY_VIDEO_DYNAMIC_ARGS% %COMFY_VIDEO_VRAM_ARGS% --reserve-vram %COMFY_VIDEO_RESERVE_VRAM% %COMFY_VIDEO_OFFLOAD_ARGS% --cache-none --disable-xformers %COMFY_VIDEO_BACKEND_ARGS% %COMFY_VIDEO_VAE_ARGS% %COMFY_VIDEO_EXTRA_ARGS%
