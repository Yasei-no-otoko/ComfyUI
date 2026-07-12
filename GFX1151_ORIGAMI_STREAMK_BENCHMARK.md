# gfx1151 Origami / Stream-K 比較

## Workgroup mapping修正

従来のgfx1151統合はTriton configからtileとoccupancyを渡す一方、`GROUP_M`を落としていた。Origamiが`GROUP_M=4`として選んだ候補が最終的にWGM 1で起動され、予測条件と実行条件が一致していなかった。

修正版は`GROUP_M`を`config_t.workgroup_mapping`へ渡し、最終選択でも保持する。対象テストは選択値/最終値が`4/1`から`4/4`となった。Anima MLP-downの該当`64x128x32`候補は約55 msから約24 msへ改善し、全候補では`128x256x64, GROUP_M=4`が23.46 ms、ATen `mm`が24.83 msだった。

修正後の1536x1536・30-step実測は74.63、75.40、75.63、76.99秒、中央値75.52秒。修正前Origami有効時の76.70秒からは回復したが、Origami無効より速いとはまだ判断しない。

## 環境

- GPU: gfx1151 (40 CU / wave32)
- PyTorch: 2.14.0a0+rocm7.15.0a20260704
- HIP: 7.15.26263
- dtype: BF16
- 形状は Anima 1536x1536 latent の主要 GEMM を模したもの

## Origami

ROCm `develop` の Origami 0.1.0をWindows対応し、PyTorch 2.14の4引数API互換を追加した。0.0.2はHIPが報告する20 WGPを20 CUとして扱っていたが、developのRDNA補正後は40 CUとして認識する。

汎用予測モデルはgfx1151で大きなtileを過大評価した。未校正版top-8はMLP-upを23.91 msから124.93 msへ悪化させたため、gfx1151 BF16に限定してwave32で実測上優位な64x128x32/128x64x32を優先した。

| GEMM (M,N,K) | Origami無効 compile | 校正済top-8 compile | 無効 median | 校正済 median |
|---|---:|---:|---:|---:|
| projection (18432,2048,2048) | 72.94 s | 4.04 s | 6.858 ms | 6.682 ms |
| MLP-up (18432,8192,2048) | 82.54 s | 3.62 s | 23.908 ms | 23.634 ms |
| MLP-down (18432,2048,8192) | 87.05 s | 3.97 s | 26.869 ms | 26.629 ms |

3形状とも実測最速帯をtop-8に残し、compile時間を約18〜23倍短縮、定常時間を約0.9〜2.6%改善した。

## Stream-K

hipBLASLtの`TENSILE_SOLUTION_SELECTION_METHOD=2`とdynamic grid 6を通常選択(method 0)と比較した。

| GEMM | method 0 | method 2 requested | 差 |
|---|---:|---:|---:|
| projection | 5.585 ms | 5.695 ms | +2.0% |
| MLP-up | 17.748 ms | 17.922 ms | +1.0% |
| MLP-down | 25.025 ms | 25.023 ms | 同等 |

Origami 0.1.0導入後に再試験しても、hipBLASLtログではmethod 0/2とも同じ通常solution index 657 (`Others`)を選択し、この形状ではgfx1151向けStream-K kernelは選ばれなかった。method 2の選択経路自体は有効だが、現行ライブラリと対象形状では通常解へフォールバックするため既定値には採用しない。Stream-K kernelを含む将来ライブラリまたは別形状で再評価する。

### 低occupancy / large-K再試験

短区間GPU eventが負値を返したため、各event内で同一GEMMを100回実行して1回当たりへ正規化した。

| (M,N,K) | method 0 | method 2 | method 2差 |
|---|---:|---:|---:|
| (256,256,8192) | 0.0989 ms | 0.0967 ms | -2.2% |
| (512,512,8192) | 0.2577 ms | 0.2568 ms | -0.4% |
| (768,768,8192) | 0.4000 ms | 0.3975 ms | -0.6% |
| (1024,768,8192) | 0.4387 ms | 0.4375 ms | -0.3% |
| (1024,1024,8192) | 0.5649 ms | 0.5614 ms | -0.6% |
| (1536,1024,4096) | 0.5158 ms | 0.5106 ms | -1.0% |

最も差が大きい256x256x8192でも両methodはsolution index 621 (`Others`)を選択した。method 2は設定として認識されるが、今回の低occupancy形状でもStream-K code objectは選択されていない。したがって小差をStream-K効果とは判定しない。

## 採用設定

- `TORCHINDUCTOR_ORIGAMI=1`
- `TORCHINDUCTOR_ORIGAMI_TOPK=8`
- `TENSILE_SOLUTION_SELECTION_METHOD=0`
- WindowsではPyTorch 2.14の`pass_fds`問題を避けるため`TORCHINDUCTOR_COMPILE_THREADS=1`

ビルド済みwheelは`user/wheels/origami-0.1.0-cp313-cp313-win_amd64.whl`、再現用差分は`patches/origami-gfx1151-windows.patch`に保存する。
