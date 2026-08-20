@echo off
setlocal
cd /d C:\ComfyUI

set "PYTHONNOUSERSITE=1"
set "COMFY_HIP_ARCHS=gfx1151"
set "PYTHONPATH=C:\ComfyUI\custom_nodes\comfy-kitchen.disabled;%PYTHONPATH%"

"C:\ComfyUI\.venv-rocm101\Scripts\python.exe" main.py ^
  --listen 0.0.0.0 ^
  --enable-dynamic-vram ^
  --vram-headroom 4 ^
  --reserve-vram 8 ^
  --disable-async-offload ^
  --disable-xformers ^
  --disable-nvml-pressure ^
  --fast-disk
  --cache-ram ^
  --use-flash-attention %*


endlocal
