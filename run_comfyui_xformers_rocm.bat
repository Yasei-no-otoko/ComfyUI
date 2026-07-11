@echo off
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
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8"
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
set "COMFYUI_ENABLE_MIOPEN=1"

"%PYTHON_EXE%" -u main.py --enable-manager --listen 0.0.0.0 --gpu-only --enable-triton-backend --disable-xformers --fast fp16_accumulation
