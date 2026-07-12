@echo off
setlocal
cd /d C:\ComfyUI

set "PYTHON_EXE=C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\python.exe"
set "PYTHON_SITE_PACKAGES=C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\Lib\site-packages"
set "PYTHONPATH=C:\ComfyUI\custom_nodes\comfy-kitchen;%PYTHONPATH%"
set "ROCM_CORE=%PYTHON_SITE_PACKAGES%\_rocm_sdk_core"
set "ROCM_DEVEL=%PYTHON_SITE_PACKAGES%\_rocm_sdk_devel"
set "ROCM_LIBS=%PYTHON_SITE_PACKAGES%\_rocm_sdk_libraries_gfx1151"

set "PATH=%ROCM_LIBS%\bin;%ROCM_CORE%\bin;%ROCM_DEVEL%\bin;%PATH%"
set "INCLUDE=%ROCM_CORE%\include;%ROCM_DEVEL%\include;%INCLUDE%"
set "LIB=%ROCM_CORE%\lib;%ROCM_DEVEL%\lib;%ROCM_LIBS%\lib;%LIB%"
set "ROCM_HOME=%ROCM_CORE%"
set "ROCM_PATH=%ROCM_CORE%"
set "HIP_PATH=%ROCM_CORE%"

rem PISA wheel and kernels are validated only for gfx1151.
set "PYTORCH_ROCM_ARCH=gfx1151"
set "GPU_TARGETS=gfx1151"
set "CMAKE_HIP_ARCHITECTURES=gfx1151"
set "HIP_ARCHITECTURE=gfx1151"

rem Keep model, CLIP, latent, and PISA working sets resident in the large shared VRAM pool.
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8"
set "COMFYUI_ENABLE_PYTORCH_VAE_ON_AMD=1"
set "COMFY_KITCHEN_SAVED_INT8_POLICY=mixed"
set "TORCH_BLAS_PREFER_HIPBLASLT=1"
set "TORCH_BLAS_PREFER_CUBLASLT=1"
set "ROCBLAS_USE_HIPBLASLT=1"
set "CUBLASLT_WORKSPACE_SIZE=262144"
set "HIPBLASLT_TUNING_USER_MAX_WORKSPACE=268435456"

rem Reuse Inductor/FlexAttention kernels and spend more compile time on steady-state GEMM quality.
set "TORCHINDUCTOR_CACHE_DIR=C:\ComfyUI\user\.cache\torchinductor_pisa_gfx1151"
set "TORCHINDUCTOR_FX_GRAPH_CACHE=1"
rem PyTorch 2.14 Windows SubprocPool passes pass_fds when threads > 1 and crashes.
rem Origami top-8 keeps the single-process compile short while avoiding that path.
set "TORCHINDUCTOR_COMPILE_THREADS=1"
set "MAX_JOBS=32"
set "TORCHINDUCTOR_MAX_AUTOTUNE=1"
set "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1"
set "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=DEFAULT"
set "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1"
rem The experimental event benchmarker returns negative timings on Windows ROCm
rem for sub-millisecond GEMMs and can select the wrong kernel.
set "TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER=0"
rem These Origami knobs become active when the optional rocm-origami package is installed.
set "TORCHINDUCTOR_ORIGAMI=1"
set "TORCHINDUCTOR_ORIGAMI_TOPK=8"

rem The current gfx1151 hipBLASLt library falls back to a normal solution when
rem method 2 is requested, so keep Stream-K disabled until a real kernel ships.
set "TENSILE_SOLUTION_SELECTION_METHOD=0"
set "TENSILE_STREAMK_DYNAMIC_GRID=6"

rem Flash remains the validated fallback for the first four and cross-attention blocks.
set "TORCH_ROCM_FA_PREFER_CK=0"
set "FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE"
set "COMFYUI_ENABLE_MIOPEN=1"
set "PYTHONFAULTHANDLER=1"
set "TQDM_MININTERVAL=1"
if exist "%ROCM_LIBS%\bin\hipdnn_plugins\engines" set "HIPDNN_PLUGIN_DIR=%ROCM_LIBS%\bin\hipdnn_plugins\engines"
if exist "%ROCM_LIBS%\bin\hipdnn_plugins\heuristics" set "HIPDNN_HEURISTIC_PLUGIN_DIR=%ROCM_LIBS%\bin\hipdnn_plugins\heuristics"

"%PYTHON_EXE%" -u main.py --enable-manager --listen 0.0.0.0 --gpu-only --cache-ram --use-flash-attention --disable-xformers --enable-triton-backend --fast autotune %*
