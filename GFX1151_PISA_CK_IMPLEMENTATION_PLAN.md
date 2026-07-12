# GFX1151向けComposable Kernel PISA 実装方針

更新日: 2026-07-12

## 0. 実装結果

本方針に基づく実装は`rdna35-pisa-ck` 0.7.0まで完了した。Windows ROCm向けpip wheelを`MAX_JOBS=32`、`gfx1151`単一targetでbuildし、Python 3.13 / PyTorch `2.14.0a0+rocm7.15.0a20260704` / ROCm `7.15.26263`へ導入した。検証済みCK commitは`4975bd0c8e17a54bdc27c746527a385e7383bb07`である。

最終構成は完全CK sparse FMHAではなく、gfx1151でcompile・実測できる次のhybridである。

1. C++/HIP + CK Tile型: block statistics
2. C++/HIP: 8 tokens x 8 heads x D128、16 KiB shared tileによるQ/K/V空間配置変換
3. PyTorch 2.14 FlexAttention: exact blocksとcentroid tail
4. WMMA FP32 accumulation: centered first-order correction
5. C++/HIP: raster順のmerged outputへ復元

BF16 `B=2,H=16,T=9216,D=128`のAPI 5最終wheel実測は、native pack 2.412 ms、PISA 23/144 blocks 37.917 ms、ComfyUI Flash Attention 46.411 msだった。PISA callはFlashより18.3%高速である。

API 5最終wheelを同一常駐プロセスでwarm-up後、同一seedのAnima INT8_ConvRot、1536x1536、Spectrum euler/simple、30 steps、CFG 5、SEA off、17 actual forwardsで比較した。FlashはSampler 69.12秒 / Prompt total 69.17秒、PISAは68.63秒 / 68.68秒で、SamplerとPrompt totalを0.49秒（0.7%）短縮した。同seed画像は破綻せず、Flash比SSIM 0.961379、RGB cosine 0.999484だった。近似差を許容する明示opt-in用途とする。

実Animaでは33/36 exact blocksが初stepから非有限化し、32 blocksは初stepが有限でも30-step途中でNaN化した。production spatial pathは実測完走した23 blocksだけをsparse profileとして許可し、0〜22および24〜143を明示エラー、144だけをdense SDPA検証pathとして許可する。ComfyUI nodeも23/144へ固定し、最初の4 transformer blocksはFlash、後段24 self-attention blocksだけをPISAへ送る。

## 1. 目的

GFX1151上のAnima画像生成で、現在のPython/Triton試作版PISA（Piecewise Sparse Attention）を実用的な速度へ引き上げる。PISA本体をComposable Kernel（以下CK）のC++/HIPカーネルとして実装し、Python 3.13・PyTorch 2.14 nightly・Windows ROCm 7.15環境へpip wheelとして導入する。

最終成果は次の3点とする。

1. `rdna35_pisa_ck`としてimportできる、GFX1151専用のバイナリwheel
2. 現行PISAの数式とComfyUI側インターフェースを保つCK実装
3. 実測に基づき、CK PISA・既存Flash Attention・既存Triton実装を安全に選択するComfyUI統合

## 2. 非目標

- CUDA/NVIDIA、Linux、ROCmの全GPU世代を同時にサポートしない。
- 学習・逆伝播は実装しない。推論forward専用とする。
- int8のQ/K/V attention演算は導入しない。INT8_ConvRotモデルの線形層出力であるbf16 Q/K/Vを高速化する。
- ComfyUI起動中のネットワーク取得やnative extensionのJITビルドは行わない。FlexAttentionの初回Inductor compileはcold latencyとして分離する。
- 品質検証を通る前にCK PISAを`auto`の既定値へ昇格しない。
- CK、AITER、PyTorchの内部実装をComfyUI coreへコピーしない。

## 3. 現在確認できている環境と資産

### 実行環境

- OS: Windows
- GPU target: `gfx1151`
- Python: `C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\python.exe`
- PyTorch: 2.14 nightly、ROCm 7.15
- ComfyUI branch: `perf/anima-rocm-int8-auto-path`

### 既存PISA実装

- ComfyUI custom node:
  `custom_nodes/ComfyUI-RDNA35-FixedBlockAttention/rdna35_block_attention/`
- PISA制御・参照実装: `pisa_attention.py`
- Triton preparation試作: `pisa_kernel.py`
- 公式PISA checkout: `C:\Users\HarutoWatanabe\piecewise-sparse-attention`

現行契約は、連続な`[BH, T, 128]`のfp16/bf16 self-attention、`block_size=64`、non-causal、forward-onlyである。この契約を最初のCK実装でも維持する。

### CK実装の参考資産

- AITER Python API: `...\site-packages\aiter\ops\mha.py`
- AITER JIT/build制御: `...\site-packages\aiter\jit\core.py`
- AITER MHA target生成: `...\site-packages\aiter\jit\utils\mha_recipes.py`
- AITERが解決するCK候補: `...\site-packages\aiter_meta\3rdparty\composable_kernel`

ただし、現在導入済みのAITER wheelには上記`3rdparty/composable_kernel`ディレクトリが実在せず、AITER自身も`Triton ops only`と判定している。したがって、AITERをruntime依存にせず、ビルド時に明示したCK source treeを利用する。

## 4. 実装上の基本判断

### 4.1 独立wheelにする

ネイティブ実装はcustom nodeのPythonコードへ直接埋め込まず、独立した`rdna35_pisa_ck`パッケージとして作る。

- build時: PyTorch headers、Windows ROCm SDK、固定したCK source treeを使用
- runtime時: PyTorch/ROCm runtime以外のAITER依存を持たない
- 配布物: `cp313-win_amd64`かつ`gfx1151`コードオブジェクトのみを含むwheel
- ComfyUI側: optional importと能力判定だけを持つ

開発用source配置は次を第一候補とする。

```text
custom_nodes/ComfyUI-RDNA35-FixedBlockAttention/
  native/rdna35_pisa_ck/
    pyproject.toml
    setup.py
    rdna35_pisa_ck/
      __init__.py
    csrc/
      bindings.cpp
      pisa_kernels.cu
    tests/
```

ビルド生成物、Ninja作業ディレクトリ、wheelはリポジトリへコミットしない。最終的に必要なsource、テスト、短いbuild手順だけを残す。

### 4.2 CK sourceの扱い

CK sourceの探索順を次に固定する。

1. build時の`CK_DIR`
2. 検証済みのローカルCK checkout
3. AITER full-source checkout内の`3rdparty/composable_kernel`

見つからない場合は、別実装へ暗黙に切り替えずbuildを明確なエラーで止める。pip build中の自動cloneやダウンロードは行わない。検証済みCK commitはbuild metadataと文書へ記録し、wheelの再現性を確保する。

### 4.3 公開APIを狭く保つ

production APIはAnimaの元layoutを保つ次の形に確定した。

```python
rdna35_pisa_ck.forward_spatial_bhtd(q, k, v, exact_blocks=23) -> torch.Tensor
```

入力は連続な`[B,T,H,D]` storageに対するBF16 `[B,H,9216,128]` view、出力はmerged raster順`[B,9216,H*128]`とする。generic `[BH,T,128]`の`forward`は数値検証・benchmark APIとして残す。production sparse profileは23 blocksだけで、144 blocksはdense SDPA検証用である。

C++境界でdevice、dtype、shape、contiguous、head dimension、block size、architectureを検証する。未対応入力を黙ってcast・copy・padせず、Pythonのdispatch層が既存backendへfallbackする。

### 4.4 検証後に確定した実装境界

gfx1151へCK sparse-attention VSAを直接移植する試験では、gfx9/gfx12向けasync global-to-LDS命令がgfx1151でcompileできなかった。scalar CKだけでPISA全体を実装した版も、BF16 `BH=32,T=4096,D=128`で約95 msとなりFlash Attentionの約9 msを大きく下回った。

このため、実用版は次のhybrid境界に固定する。

1. CK Tile C++/HIP: Q/K centroidとV meanを1 kernelで計算
2. PyTorch 2.14: GPU上でrouting/top-kとcompact block indexを生成
3. FlexAttention: exact token blocksとlength-weighted centroid tailを別partitionとして計算し、LSEで共通softmaxへ統合
4. WMMA BMM: global first-order correctionをfp32 accumulationで適用

初期試作ではfull-forwardのopaque custom opから内部Flex compileを呼ぶと、FlexAttentionが暗黙生成するfull maskと外側AOTAutogradが衝突した。v0.4ではdeviceごとに最小のfull `BlockMask`を`prepare()`で事前生成し、approximate partitionへ明示的に渡すことでこの衝突を解消した。PISA全体をfake implementation付きのopaque custom opとして外側Inductorへ公開し、内部で一度compileしたFlex kernelを全layer/stepで再利用する。これによりComfyUIのmodel graphをPISA layerごとに分断せず、`T=9216`の外側fullgraph compileは約7.2秒で完了した。

連続`[BH,T,D]`だけを測ったv0.4ではlayout変換コストが含まれていなかった。実モデルの`[B,T,H,D]` storageから測るとPython layout copyがFlashとの差を消したため、v0.5でspatial pack/unpackをC++/HIPへ移した。最初のhead固定packは19.54 msだったが、8 tokens x 8 headsのshared transposeで2.412 msまで短縮した。最終v0.7.0 API 5の完全spatial callは23/144 blocksで37.917 ms、Flash Attentionは46.411 msである。

実モデルのcompute dtypeはBF16であり、FP16はblock sumとfirst-order correctionで有限入力からoverflowし得ることを確認した。CK PISAのproduction契約はBF16専用とし、FP16は既存Flash backendへ戻す。exact budgetは多いほど常に安定するとは限らず、実Animaで32/33/36 blocksの非有限化を確認したため、完走確認済みの23 blocksだけをproduction sparse上限とする。

## 5. PISA数式の実装対応

現行参照実装の意味を変更せず、次の3段階へ分解する。Stage AをCK C++/HIP、Stage B/CをPyTorch 2.14のGPU graphで実装する。

### Stage A: block statistics preparation

各key blockについてfp32で以下を計算する。

- `K_mean[b,h,kb,d]`: block内Kの平均
- `V_mean[b,h,kb,d]`: block内Vの平均
- 末尾blockの有効token数

CK版では64-token block、D=128を128-thread workgroupで処理し、各入力を`1 / length`倍してfp32 accumulatorへ加える。first-order項は`H_sum = Σ_block ((K - K_mean)^T V)`としてKを先にcenter化し、巨大なGEMM結果同士の減算を避けてPyTorch WMMA BMMで生成する。

### Stage B: query routing

- query blockごとに`Q_mean`を計算
- `Q_mean @ K_mean^T * scale`をPyTorch WMMA BMMで計算
- query blockごとに`exact_blocks`個のkey blockを選択
- `sink_block`指定時は必ず選択集合へ含める

選択結果は`[BH, NB, exact_blocks]`のcompact indexとしてFlexAttentionの`BlockMask`へ渡す。未選択tailをscore modifierで除外するため`[BH,NB,NB]` bool selectionも一時的に使うが、executionを越えて保持せず、token-level QK matrixは生成しない。

### Stage C: exact blocksとapproximate tailの統合

query block単位で以下を同じonline-softmax状態へ統合する。

1. 選択されたkey blockはtoken-level `QK^T`と`P@V`をFlexAttentionで正確に計算
2. 非選択blockは`Q @ K_mean`、`V_mean`、block lengthで0次近似
3. global first-order correctionとして`Q @ H_sum * scale / T`を適用
4. exact/approximateの双方で共有するmax、denominator、numeratorをfp32で更新
5. 最終結果だけを入力dtypeへ変換

exact partitionとapproximate partitionはPyTorch 2.14 FlexAttentionで計算し、それぞれのnatural-log LSEを`logaddexp`で統合する。blockごとのPython loopやdense attentionの反復起動は行わない。

## 6. GFX1151向けカーネル設計

### 初期固定条件

- architecture: `gfx1151`
- dtype: bf16
- head dimension: 128
- PISA block: 64 tokens
- attention: non-causal self-attention
- production input layout: contiguous `[B,T,H,D]` storageの`[B,H,T,D]` view
- accumulation: fp32
- sequence length: 64以上、末尾非整列をmask処理

### 探索する実装候補

- wave size: 32 / 64
- query tile: 32 / 64 tokens
- key tile: 64 tokens
- GEMM instruction/layout: CKがgfx1151向けに生成可能な候補だけを使用
- pipeline stages: 1 / 2 / 3
- waves per workgroup: 2 / 4 / 8
- exact block indexのshared memory配置
- `H_sum`のatomic版 / workspace reduction版

候補数を無制限に増やさない。Animaで実際に現れる`BH,T,D,dtype`をログから収集し、そのshape集合だけでoffline benchmarkして最小のdispatch tableを生成する。

### launch構成

1. CK Q/K/V block statistics
2. GPU routing/top-k
3. Flex exact partition
4. Flex approximate partition
5. WMMA first-order correctionとLSE合成

各項目はblock数に依存するPython loopを持たない。将来CK Tileがgfx1151向けnon-async sparse FMHA pipelineを提供した場合に限り、Stage B/Cの完全native融合を再評価する。

## 7. PyTorch・Python binding

- `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL`によるcustom op登録を第一候補とする。
- pybind moduleはimportとversion/capability照会に限定する。
- current PyTorch stream上でlaunchし、内部でdevice synchronizeしない。
- allocatorはPyTorch tensor/workspaceを使い、独自の永続GPU cacheを持たない。
- executionを越えて大きなtensorを保持しない。
- autograd kernelは登録せず、requires-grad入力は拒否する。
- `torch.compile`から呼ばれる可能性に備え、fake/meta implementationはPython package側で必要最小限提供する。

ABIは実際にComfyUIを動かすPython/PyTorchへ合わせる。別PyTorch buildで作ったwheelを流用せず、wheel metadataにPyTorch/ROCm/CK/gfx target情報を記録する。

## 8. Windows ROCm build方針

### 必須条件

- `--offload-arch=gfx1151`以外のGPU code objectを作らない。
- `MAX_JOBS=32`を設定する。
- build logで実際に32並列が使用されたことを確認する。
- Python 3.13の実行環境からwheelをbuildする。
- ROCm SDKのsplit packageからinclude、lib、binを解決する。

build backendはまず`setuptools`とPyTorchのextension build機構を使い、Windows ROCmで必要なhipcc/clang invocationをAITERの既存build処理と照合する。CMake導入はPyTorch extension buildではCK translation unitを正しく生成できないことを実証した場合に限る。

最適化flagは一度に増やさず、まず正しい`gfx1151` code objectを作る。その後、CKの推奨flagと生成ISAを確認し、fast-mathがPISAのsoftmax安定性・品質を損なわない範囲だけで適用する。

## 9. 検証ゲート

### Gate 1: package/build

- clean buildでwheelが生成される
- clean Python processでimportできる
- binaryが`gfx1151` code objectを含む
- ComfyUI起動時にJIT compileしない
- repositoryにbuild生成物や一時ファイルが残らない

### Gate 2: 数値一致

BF16について、少なくとも`T=64,65,128,512,4096,9216`を検証する。

- Stage AをPython fp32 referenceと比較
- routing indexを同一score/tie policyで比較
- `exact_budget=1.0`をPyTorch SDPAと比較
- approximate PISAを現行Python PISA referenceと比較
- NaN/Inf、末尾block、`sink_block`、`exact_blocks=0/1/NB`を検証

許容誤差はdtypeとshape別に実測して固定する。cosine similarityだけで合格にせず、max absolute error、mean absolute error、relative errorも保存する。

### Gate 3: 合成Q/K/V性能

benchmarkはwarmup後に`torch.cuda.Event`で測り、各caseのmedianとp95を記録する。compile/import時間はsteady-stateと分離する。

既存の参考値:

| Shape | 既存Flash BF16 |
|---|---:|
| B2/H16/D128, T=4096 self | 9.376 ms |
| B2/H16/D128, T=4096 x 512 cross | 1.763 ms |
| B2/H16/D128, T=9216 self | 48.137 ms |

PISA対象はself-attentionのみとし、cross-attentionへ誤dispatchしない。実用候補へ昇格する条件は、品質差を許容するbudgetでT=9216の完全spatial callがFlashより速く、同一forward数の30-step Prompt totalも短縮し、未対応shapeをFlashへ残すこととする。20% attention短縮は目標値とし、実測18.3%でもend-to-end短縮が再現できれば明示opt-in候補とする。

### Gate 4: 実Anima Q/K/V品質

合成random tensorだけでbudgetを決めない。Anima-Turbo-int8convrotの実attention層から、1回のexecution内だけで検証用Q/K/V統計を取得し、次を比較する。

- CK PISA対Python PISA
- CK PISA対Flash/dense attention
- layerごとのoutput cosine/error
- timestepと解像度による品質変化
- exact budgetごとの速度・誤差曲線

検証tensorを永続保存する場合はユーザーが明示したbenchmark artifactだけとし、通常生成pathにcapture機能を残さない。

### Gate 5: ComfyUI end-to-end

- 同一seed、workflow、checkpoint、8/30 stepsで比較
- samplerだけでなくPrompt totalをwall-clock計測
- 初回compileを除いた2回目以降を主指標にする
- VRAM peak、model load、VAE decode、image saveを別区間として記録
- 画像差分を確認し、速度表示だけの改善を成功扱いしない

## 10. ComfyUI統合とfallback

custom nodeのdispatch順は、検証完了後に限り次のようにする。

1. 明示的にCK PISAを選択し、対応shapeなら`rdna35_pisa_ck.forward`
2. `auto`かつ実測dispatch tableでCK PISAが優位ならCK PISA
3. それ以外は既存Flash Attention
4. Flashが利用不能な場合だけ既存の安全なComfyUI attention backend

fallback条件はPython側で明示し、C++例外発生後に毎step再試行しない。architecture、dtype、shape、self/cross、exact budgetの能力を一度判定し、診断ログは短く1回だけ出す。

既存のmodel patcher経由でattentionを差し替え、nodeからモデル内部を直接patchしない。選択済みattention callableの関数名やmodule名を上位層が検査する実装にも戻さない。

## 11. INT8_ConvRotとの関係

INT8_ConvRotはweight storage/GEMM pathであり、attentionへ入るQ/K/Vはbf16またはfp16である。したがってCK PISAは次を守る。

- model loaderが選択したcompute dtypeを維持
- Q/K/Vをint8へ再量子化しない
- bf16/fp16をshapeごとにbenchmarkし、速いdtypeはmodel-management層で選ぶ
- attention内部で不要なfp32 tensor全体を作らず、accumulatorだけfp32にする

これにより、第2 MLPではbf16がINT8より速いという既存結果と矛盾せず、各演算を最速の実装へ個別dispatchできる。

## 12. 実装順序

1. ローカルCK source、commit、Windows ROCm SDK pathを確定
2. package skeletonと完全ABI/capability checkを作成
3. CK block statisticsを実装し、Python fp32 reference一致を達成
4. routing/top-k、Flex exact/tail、first-order correctionを統合
5. `exact_budget=1.0`でSDPA、partial budgetでPython PISA reference一致を達成
6. 32並列でwheel build、install、clean-process smoke test
7. 合成Anima shapeでFlash/PISAのdispatch境界を確定
8. model ownerからself-attention markerを渡し、model-local patchを統合
9. 同一seedの実Anima 30-stepでquality/Prompt totalを比較
10. 不要な試作・build artifact・`__pycache__`を削除し、テスト後にcommit/push

各段階で前段のcorrectness gateを通してから次へ進む。性能が出ない段階は数値一致する最小実装を保持して原因を計測し、根拠なく融合範囲を広げない。

## 13. 主なリスクと対処

### CK sourceが現在のAITER wheelにない

検証済みローカルcheckoutを明示し、commitを固定する。runtime AITER依存にはしない。

### CK upstreamのgfx1151 FMHA候補が生成されない

AITERの`--targets gfx1151`生成経路とCK Tile primitiveを確認する。dense FMHA instanceをそのまま利用できなくても、gfx1151でcompile可能なCK Tile GEMM/softmax primitiveからPISA専用kernelを構成する。

### sparse化してもrouting/launch overheadで遅い

bool maskとPython block loopを禁止し、compact index、GPU top-k、2〜4 launchで実装する。Tが短いshapeはFlashへ戻す。

### 近似誤差が生成品質を落とす

実Q/K/Vと30-step生成で安全域を決める。今回のproduction profileは23/144 blocksへ固定し、構図差を明記した明示opt-inとする。未検証budgetをUIから選べる状態にせず、32/33/36 blocksの非有限化を受けて0〜22および24〜143 blocksを実行前に拒否する。

### wheel ABIがnightly PyTorch更新で壊れる

torch buildごとにwheelを作り直し、import時にbuild metadataを照合する。ABI不一致時は明確に無効化してFlashへfallbackする。

## 14. 完了条件

次の全項目を満たした時点で「GFX1151向けCK PISAが実用レベル」と判断する。

- CK C++/HIP block-statistics sourceとPyTorch 2.14 FlexAttention合成からpip wheelを32並列で再現buildできる
- Python 3.13 / 現行torch 2.14 nightly / ROCm 7.15 / gfx1151でimport・実行できる
- exact pathがSDPA、approximate pathがPython PISA referenceと許容誤差内で一致する
- FP16と異なるPyTorch/ROCm nightlyを実行前に拒否し、既存Flash pathへ残す
- 実Anima Q/K/Vで採用budgetの品質基準を満たす
- 対象self-attention shapeでFlashより速く、同一forward数のPrompt totalも短縮する
- ComfyUI 8/30-stepのPrompt totalが再現可能に短縮する
- 未対応shape・dtype・GPUでは既存backendへ安全にfallbackする
- build artifact、dead code、debug capture、永続tensor cacheを残さない
- 変更をテスト後に新規commitとしてpushする

上記性能条件に届かない場合、CK packageは研究backendとして明示選択時だけ利用可能にし、`auto`へは組み込まない。速度表示ではなく、GPU eventとPrompt wall-clockの双方で採否を決定する。
