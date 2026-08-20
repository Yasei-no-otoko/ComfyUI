# Python 3.14 Free-threaded Python と Windows ROCm の移行判断

調査日: 2026-07-11

## 結論

**現時点では Python 3.14 の Free-threaded build (`3.14t`) へ移行しない。**

Free-threading は複数の CPU-bound Python スレッドを並列実行できるようにする機能で、単一 GPU の拡散推論カーネル自体を速くするものではない。現在の ComfyUI は `cuda:0` の AMD Radeon 8060S (gfx1151) で Flash Attention と ROCm 7.15 を使っており、実行時間の大半は GPU 推論である。したがって、対応していたとしても end-to-end の画像生成が速くなる見込みは低い。

さらに、現在の Windows ROCm 配布物にはこの環境に必要な `cp314t` wheel がない。AMD の gfx1151 device-wheel index には同じ Torch 2.14 / ROCm 7.15 日付の `cp314-cp314-win_amd64` はあるが、`cp314t` はない。Free-threaded Python では通常 ABI の extension wheel を再利用できないため、現行環境を 3.14t に切り替えると ROCm GPU 経路を維持できない。

Python 3.14 Free-threading は CPU の明示的な並列処理には有用になり得るが、ComfyUI の単一 GPU 推論を速くするための移行先ではない。CPython 自身も単一スレッドの Python コードに 5–10% のオーバーヘッドがあり得るとしています。

## この環境で確認した事実

| 項目 | 現状 | 判断への影響 |
| --- | --- | --- |
| 実ランタイム | Python 3.13.9、GIL 有効、Torch `2.14.0a0+rocm7.15.0a20260704`、ROCm 7.15 | 現在の高速経路は 3.13 ABI 用 |
| Python 3.14 | `Python314` ディレクトリはあるが実行ファイルがなく、`py -0p` にも 3.14 / 3.14t が未登録 | まず正式な 3.14t runtime の追加が必要 |
| Torch | `torch._C.cp313-win_amd64.pyd` | 3.14t では再利用不可 |
| 高速化拡張 | `triton-windows`、`xformers`、`flash_attn_2_cuda.cp313-win_amd64.pyd`、`comfy-kitchen`、`ComfyUI-FeatherOps` の native extension がある | すべて `cp314t` wheel または free-threading 対応ソース build が必要 |
| AMD gfx1151 wheel | `cp314` Windows wheel はあるが `cp314t` は未公開 | 現時点の決定的な blocker |
| 実行構造 | `comfy.multigpu.MultiGPUThreadPool` は複数 GPU 用。現在は単一 GPU | GIL を外しても単一 GPU の diffusion step は並列化されない |

`comfy-kitchen` の CUDA backend は `abi3` wheel だが、free-threaded CPython は stable/limited ABI をサポートしないため、これも「そのまま使える」根拠にはならない。

## 何が速くなり得るか

次の条件を満たすワークロードに限り、3.14t の価値を測定する余地がある。

- 複数の CPU-bound custom node を、共有状態を持たず複数 `threading.Thread` で並列実行する場合
- 複数の独立した CPU 前処理、画像エンコード、ハッシュ計算を明示的に並列化した場合
- 将来、複数 GPU で GPU ごとの Python worker が CPU-bound な準備を競合なく行う場合

次は速くならない、または遅くなる可能性が高い。

- 単一 GPU の UNet / Transformer / VAE / attention 実行
- 現在の 1 prompt を 1 GPU で処理する通常の ComfyUI workflow
- GIL 非対応の native extension を import して、実行時に GIL が再有効化された場合
- Python オブジェクトやメモリ使用量が多い workflow。3.14t は通常 build よりメモリ使用量が増え得る

## Go / No-go 判定ゲート

以下を**すべて**満たすまでは、現在の `run_comfyui_int8_convrot_rocm.bat` を置き換えない。

1. Windows に正式な Python 3.14t が導入され、`py -3.14t -VV` が成功する。
2. AMD の gfx1151 対応 wheel index に、同じ release set の `cp314-cp314t-win_amd64` がある。
   - `torch`, `torchvision`, `torchaudio`
   - `amd-torch-device-gfx1151`, `amd-torch-device-gfx115x`
   - 必要な ROCm SDK/device package
3. `triton-windows`、xFormers、Flash Attention、comfy-kitchen、FeatherOps を含む全 native extension に 3.14t build がある、または source build が実証済みである。
4. すべて import した後も `sys._is_gil_enabled()` が `False` である。
5. 同一 workflow の warm run で、画像の整合性を保ちながら end-to-end median が現行 Python 3.13 より少なくとも 5% 改善する。

AMD wheel index の確認例:

```powershell
Invoke-WebRequest https://rocm.nightlies.amd.com/whl-multi-arch/amd-torch-device-gfx1151/ |
    Select-String 'cp314-cp314t-win_amd64'
```

出力が空なら No-go である。`cp314` は通常の 3.14 ABI であり、`cp314t` とは別物である。

## 将来の可逆的な移行パス

### 1. 現行環境を保持する

- 現行の Python 3.13 launcher、ROCm SDK、site-packages、model directory を変更しない。
- 新環境は `C:\ComfyUI\.venv-py314t` とし、Python 3.13 の `site-packages` をコピーしない。
- 移行中も既存の `run_comfyui_int8_convrot_rocm.bat` を常に rollback launcher として残す。

### 2. Python 3.14t を正しく導入する

Python for Windows installer で **Customize installation** → **Download free-threaded binaries** を選ぶ。追加される executable は `python3.14t.exe` で、`py -3.14t` から選択できる。

```powershell
py -3.14t -VV
py -3.14t -c "import sys, sysconfig; print(sys.version); print(sysconfig.get_config_var('Py_GIL_DISABLED')); print(sys._is_gil_enabled())"
```

期待値は `Py_GIL_DISABLED == 1` と `sys._is_gil_enabled() == False` である。

### 3. 隔離した 3.14t virtual environment を作る

```powershell
$py314t = 'C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python314\python3.14t.exe'
& $py314t -m venv C:\ComfyUI\.venv-py314t
$venvPy = 'C:\ComfyUI\.venv-py314t\Scripts\python.exe'
& $venvPy -m pip install --upgrade pip
```

この段階ではまだ `requirements.txt` を一括 install しない。先に ROCm GPU stack の `cp314t` wheel が解決できることを確認する。

### 4. GPU stack を dry-run で検証する

AMD が `cp314t` wheel を公開した後、その release の公式 install command を `--dry-run` で実行する。解決結果に通常 ABI の `cp314-cp314-win_amd64`、source fallback、CPU-only torch が混ざった場合は中止する。

```powershell
& $venvPy -m pip install --dry-run --pre <AMDが公開したcp314t用の完全なtorch/ROCm依存セット>
```

次に、残る依存を 3.14t 対応版だけで導入する。source build が必要な native extension は、Windows では build backend に `Py_GIL_DISABLED=1` を明示し、各 extension が `Py_MOD_GIL_NOT_USED` 相当を宣言していることまで確認する。単に `/DPy_GIL_DISABLED=1` を渡すだけでは thread-safe にはならない。

### 5. import 後に GIL が再有効化されていないことを確認する

```powershell
@'
import sys
import sysconfig

assert sysconfig.get_config_var('Py_GIL_DISABLED') == 1
assert sys._is_gil_enabled() is False

import torch
import triton
import xformers
import flash_attn
import comfy_kitchen

assert torch.cuda.is_available()
assert torch.version.hip is not None
assert sys._is_gil_enabled() is False, 'A native extension re-enabled the GIL'
print(torch.__version__, torch.version.hip)
'@ | & $venvPy -
```

ここで GIL が再有効化された場合、3.14t 化による CPU 並列化の利点は失われる。launcher を作らず Python 3.13 を継続する。

### 6. 合格後だけ別 launcher を追加する

新 launcher は既存ファイルを変更せず、`run_comfyui_int8_convrot_rocm_py314t.bat` として追加する。以下は合格後のテンプレートである。

```bat
@echo off
setlocal
cd /d C:\ComfyUI

set "PYTHON_EXE=C:\ComfyUI\.venv-py314t\Scripts\python.exe"
set "PYTHON_SITE_PACKAGES=C:\ComfyUI\.venv-py314t\Lib\site-packages"
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

set "TORCH_BLAS_PREFER_HIPBLASLT=1"
set "ROCBLAS_USE_HIPBLASLT=1"
set "HIPBLASLT_TUNING_USER_MAX_WORKSPACE=268435456"
set "COMFYUI_ENABLE_MIOPEN=1"
set "FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE"

"%PYTHON_EXE%" -u main.py --enable-manager --listen 0.0.0.0 --gpu-only --use-flash-attention --disable-xformers --enable-triton-backend --fast autotune %*
```

この launcher は `cp314t` GPU stack と GIL-disabled import check が実証されるまで作成・実行しない。

### 7. ベンチマークと採否

1. 同一モデル、seed、workflow、解像度で Python 3.13 と 3.14t を比較する。
2. 各環境で warm run を捨てた後に最低 5 回測定し、median / p95 / peak VRAM / RAM を記録する。
3. GPU 推論だけでなく、画像読み込み、custom node CPU 処理、保存までを含む prompt 全体を測定する。
4. 3.14t が end-to-end median で 5% 以上速く、p95・画質・VRAM/RAM・連続実行の安定性を悪化させない場合だけ採用する。
5. 失敗時は 3.14t venv と新 launcher のみを破棄し、既存 Python 3.13 launcher を使う。

## 参照

- [Python 3.14: free-threaded Python の動作・GIL 再有効化・単一スレッドのオーバーヘッド](https://docs.python.org/3.14/howto/free-threading-python.html)
- [Windows での free-threaded binaries の導入方法と `python3.14t.exe`](https://docs.python.org/3/using/windows.html#installing-free-threaded-binaries)
- [Free-threaded C extension の ABI・Windows build 要件](https://docs.python.org/3.14/howto/free-threading-extensions.html)
- [AMD ROCm nightly: gfx1151 device wheel index](https://rocm.nightlies.amd.com/whl-multi-arch/amd-torch-device-gfx1151/)
- [AMD: Windows ROCm/PyTorch support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html)
