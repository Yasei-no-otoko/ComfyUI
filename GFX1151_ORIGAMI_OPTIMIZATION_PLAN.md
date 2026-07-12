# GFX1151向けOrigami最適化計画

## 追加検証結果（2026-07-12）

- 現行`rocm-libraries/develop`から、MSVCおよびWindows分割ROCm SDK対応を加えた`origami 0.1.0`を32並列でビルド・導入した。
- Tritonの`GROUP_M`を`config_t.workgroup_mapping`へ渡し、gfx1151のGEMM順位付けと最終workgroup mappingの両方で保持した。対象テストは選択値/最終値が従来の`4/1`から`4/4`になった。
- Anima MLP-down `(18432, 2048, 8192)` BF16では、誤ったmappingで約55 msだった`64x128x32`候補が約24 msまで改善した。全候補比較では`128x256x64, GROUP_M=4`が23.46 msで、ATen `mm`の24.83 msを上回った。
- 1536x1536・30-step Spectrum/PISAは74.63、75.40、75.63、76.99秒、中央値75.52秒。修正前Origami有効時の中央値76.70秒から回復したが、Origami無効時の2-run中央値75.21秒に対する有意な定常高速化は未確認。
- 既報のcold compileは542.95秒から287.09秒へ短縮した。現段階の確実な効果は、初回候補削減と重大なWGM不一致の解消である。
- `TENSILE_SOLUTION_SELECTION_METHOD=2`は既定化しない。低occupancy形状でもmethod 0と同じ非Stream-K solutionが選ばれ、ワークフロー中央値は80.36秒へ悪化した。
- Windows ROCmのexperimental event benchmarkerはsub-ms GEMMで負値を返したため、`TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER=0`を維持する。

### AMD公式PR候補の整理

公式差分から、単一MLP形状だけを根拠にしたgfx1151 BF16 tile係数を除外した。PR候補は以下の再現可能な構造修正だけに限定する。

- Windows分割ROCm SDKの`hip::host` imported targetと入力パス検証
- wheel build時にsource treeへ`.pyd`を書き込まない`SKBUILD`分岐
- MSVCで未対応の`__builtin_ctzll`とlambda captureの移植
- PyTorch 2.14が利用する4引数`select_topk_configs` overload
- Triton `GROUP_M`から`config_t.workgroup_mapping`への伝搬
- gfx1151で選択済みWGMをrankingとfinal mappingの両方に保持
- gfx1151の選択WGM/最終WGMが`4/4`になる回帰テスト

この候補から作成したWindows wheelはMSBuild `/m:32`で成功した。SHA256は`CA71245B7AB554CF63364BADE4139F303AF01EF52D6EE41FD5BC7A6B8693C976`。導入後のgfx1151実機テストでも選択WGM 4、最終WGM 4を確認した。

`rank_configs`既存テストはgfx1151で7338候補中、上位heuristic tie-break範囲をlatency以外で並べ替えるため、厳密なlatency昇順assertと矛盾する。これは今回の差分によるNaN/Infではなく、7338件すべて有限値で最初の逆転が57,729,164.82→57,705,650.18だった。PRでは今回のWGM回帰テストとWindows wheel buildを主検証とし、この既存テスト矛盾は別issue候補として扱う。

## 現在の検証状況（2026-07-12）

この文書は参照会話「Origami最適化計画」を実行計画として保存し、現在のWindows ROCm / gfx1151検証結果を統合したものです。

### 確認済み

- ROCm `develop`のOrigamiはgfx1151をFunctionalとして実装済み。単純移植ではなく予測順位・resource model・candidate spaceの最適化が必要。
- PyPI `rocm-origami 0.0.2`は実機を20 CUとして扱った。`develop`のRDNA WGP→物理CU補正後は40 CU。
- Windows split ROCm SDK向けCMake、MSVC非互換、PyTorch 2.14の4引数`select_topk_configs`互換を追加し、Origami 0.1.0 wheelを32並列でビルド・導入済み。
- 汎用Origami top-8はAnima MLP-up `(M,N,K)=(18432,8192,2048)`で実測最速候補を落とし、23.91 msから124.93 msへ悪化した。
- gfx1151 BF16 wave32向けに`64x128x32` / `128x64x32`を優先すると、主要3形状でcompileを約18〜23倍短縮し、定常GEMMも約0.9〜2.6%改善した。
- hipBLASLt method 2は受理されるが、Anima MLP-upではmethod 0/2ともsolution 657 (`Others`)で、17.748 ms対17.922 ms。Stream-K kernelは未選択。
- 0.1〜0.6 msの低occupancy測定ではWindows ROCm GPU eventが負値を返したため、event内100回反復へ変更して再測定中。

### 現在の差分と証跡

- 実装方針: `GFX1151_ORIGAMI_STREAMK_IMPLEMENTATION_PLAN.md`
- benchmark: `GFX1151_ORIGAMI_STREAMK_BENCHMARK.md`
- Windows/gfx1151暫定patch: `patches/origami-gfx1151-windows.patch`
- 再現wheel: `user/wheels/origami-0.1.0-cp313-cp313-win_amd64.whl`

---

## 1. 前提の修正

現行`develop`のOrigamiはGFX1151へ未移植という状態ではない。GFX1151はすでに機能対象であり、アーキテクチャ判定、RDNAのWGP→物理CU換算、GFX1151用ハードウェア定数、WMMA命令表が実装されている。一方、READMEではGFX1151は「Functional」のみで「Optimized」ではなく、最適化済みはgfx942とgfx950である。

したがって、この作業はCDNA実装の単純移植ではなく、GFX1151上での予測モデル再同定、実行資源モデル修正、候補カーネル空間最適化である。主対象はGEMM kernel自体ではなく、Origamiによるkernel configurationの順位付けと選択である。絶対レイテンシ誤差より、実測最速kernelを上位へ置けるかを重視する。

## 2. 最適化目標

問題`p`、候補`c`、実測時間`T(p,c)`に対し、実測oracleを

```text
c*(p) = argmin_c T(p,c)
```

Origami選択を`ĉ(p)`、selection regretを

```text
R(p) = T(p,ĉ(p)) / T(p,c*(p))
```

とする。目的関数はpairwise rankingとregretを中心にする。

| 指標 | 初期ゲート |
|---|---:|
| 数値正当性 | 全候補で既存基準を通過 |
| 幾何平均regret | 1.03以下 |
| 中央値regret | 1.02以下 |
| p95 regret | 1.10以下 |
| regret 1.25超 | 0.5%未満 |
| Top-5 oracle recall | 95%以上 |
| selector実行時間 | 現行比+10%以内 |
| gfx942/gfx950回帰 | 幾何平均0.5%以内 |
| 定常測定CV | 2%以下 |

## 3. 優先的に直す構造問題

### 3.1 Occupancy semantics

`config_t::occupancy`はresident wavefront数と説明される一方、Tensileの`CUOccupancy`とStream-Kではactive workgroup数として使用される。RocRoller候補は`.occupancy = 1`固定であり、resource pressureがメモリ係数や経験則へ誤吸収される。

```text
occupancy（廃止または後方互換）
active_workgroups_per_cu
waves_per_workgroup
wavefront_size
vgpr_count
sgpr_count
lds_bytes_per_workgroup
threads_per_workgroup
```

を明示する。gfx1151固有のphysical VGPR構成を近隣gfx115xと一括近似しない。

### 3.2 wave64固定値の除去

共通heuristicのepilogue/PostGSUにあるwave64前提をkernel resource metadataから得たwave32/wave64へ置換する。今回の実測ではgfx1151 BF16の最速Triton候補がwave32相当の小型長方形tileであり、汎用モデルの大型tile過大評価が重大回帰を生んだ。

### 3.3 architecture/device/kernel profileの分離

| 層 | 内容 |
|---|---|
| `architecture_profile` | WGP/CU/SIMD、命令throughput、cache line、register allocation単位 |
| `device_profile` | 実CU数、L2/MALL容量、メモリ帯域、SKU、clock、power mode |
| `kernel_profile` | wave size、VGPR/SGPR/LDS、active WG/CU、vector width、cache hint |

`hipDeviceProp_t::regsPerBlock * 4`を物理register file全体の代用にせず、rocISA capabilityとcode object resource metadataを正とする。

### 3.4 統合GPU固有条件

gfx1151はCPUと物理メモリを共有するAPUである。実験manifestへ以下を保存する。

- GPU SKU（8060S/8050S/8040S）
- LPDDR5X容量・速度・channel
- BIOS/UMA、cTDP、電源mode
- GPU/memory clock、温度
- ROCm、LLVM、kernel、firmware
- display接続、CPU memory load
- GTT/TTM、warm/cold cache

8060S以外をholdoutに含め、単一SKUへ過適合しない。

### 3.5 WGM整合性

Origamiの`predict_workgroup_mapping()`と実backend kernelのWGMを両方記録し一致させる。gfx1151は`NUM_XCD=1`なのでCDNA向けXCD mappingより、単一dieのL2 locality、WGP配置、M/N traversalを直接扱う。

### 3.6 Stream-K再同定

以下の固定値・経験則をgfx1151で再測定する。

- `MinItersPerCU = 8`
- 128 MiB workspace上限
- 128-byte cache line
- parallel reduction閾値
- reduction cost係数
- gfx950由来hybrid mode閾値

static、dynamic、data-parallel、split-Kを低occupancy・large-K・skinny形状で比較し、method 2が実際にStream-K solutionを選んだことをsolution tag/indexで確認する。

## 4. 実行計画

### Phase 0: 対象と環境固定（進行中）

1. GEMM selectorを第1対象とする。
2. FP16/BF16を優先し、INT8/INT4は第2波。
3. rocm-libraries commit、ROCm、LLVM、firmwareを固定。
4. power、temperature、CPU/display負荷を記録。
5. active deviceをdevice 0固定にしない。

完了ゲートは大規模定常GEMMのCV 2%以下、manifestによる再現、model deviceとexecution deviceの一致。

### Phase 1: 現行モデルbaseline

合法な全候補について以下を保存する。

```text
problem_id, M/N/K/batch, dtype, transpose, strides, alignment
solution index, MT/MI/DepthU, wavefront, waves/WG
VGPR, SGPR, LDS, active WG/CU, WGM, cache hints
Stream-K/split-K/reduction
measured latency, predicted latency, Origami component breakdown
clock, temperature, memory state
```

dtype、transpose、square/skinny/small-M/small-N/large-K、batch、tile、cache hint、Stream-K、warm/cold、SKUでregretを分解する。

### Phase 2: gfx1151 microbench

Compute:

- F16/BF16/I8/I4 WMMA dependent latencyとthroughput
- wave32/wave64、pipeline saturation、accumulator readback
- prologue/epilogue、vector store、bounds check
- barrier/synchronization、kernel launch

Memory:

- L0/L1相当、L2、MALL、DRAM
- read/write/RMW、stride/alignment/vector width
- A/B reuse、cache hint、active WG scaling、CPU競合

Resources:

- VGPR/SGPR/LDS境界
- threads/WG、waves/WG、active WG/CU、WGP配置

`rocprofv3 --list-avail`で実機counterを確認し、counterは原因分析、timingを正とする。

### Phase 3: resource model

`hardware_t`へWGP/CU/SIMD、LDS、VGPR/SGPR、cache line、transaction、L2 domain、実測bandwidthを追加する。`config_t`へ次を追加する。

```cpp
struct resource_usage_t {
    int wavefront_size;
    int threads_per_workgroup;
    int waves_per_workgroup;
    int active_workgroups_per_cu;
    int vgpr_count;
    int sgpr_count;
    size_t lds_bytes;
};
```

Tensile/RocRollerからresource metadataを渡し、`.occupancy = 1`とdevice 0固定を廃止する。

### Phase 4: 分析モデル再fit

fit順序を固定する。

1. WMMA latency/throughput
2. LDS bandwidth/barrier
3. memory hierarchy bandwidth
4. active WGP/WG scaling
5. prologue/epilogue
6. split-K/reduction
7. 残差heuristic

```text
L = α * pairwise ranking loss
  + β * log-latency error
  + γ * catastrophic-regret penalty
  + λ * regularization
```

shape family単位でtrain/validation/holdoutを分け、別SKU・別ROCm buildをholdoutにする。

### Phase 5: candidate space

- 小型・長方形・M/N非対称MT
- DepthU、wave32/64、waves/WG
- resource-valid occupancy variants
- vector width、DirectToLDS/VGPR、WGM
- A/B cache hint全合法組合せ
- data-parallel/Stream-K/split-K/reduction/epilogue

全展開せず、ISA/resource/alignment/backend/workspaceで静的pruneした後、Origami上位N件だけを実行候補へ渡す。current-candidate oracleとexpanded-candidate oracleを比較し、差が1%未満ならmodel、3%以上ならcandidate generationを優先する。

### Phase 6: WGM/Stream-K/後段heuristic統合

Origami ranking後のK閾値、tile/CU判定、`itersPerTile >= 16`、dtype/tile除外規則がmodel結果を暗黙変更しないようにする。Origamiが一括順位付けするか、後段規則を明示constraintとして候補生成前へ移す。

### Phase 7: CIと段階release

profile version/feature flagでbaselineとv1を比較可能にする。PRを次の順に分割する。

1. instrumentationとdata schema
2. occupancy/wave/resource semantics
3. gfx1151 hardware profile
4. candidate space
5. WGM/Stream-K
6. CI/docs/Optimized表記

CIはhardware extraction、WGP→CU、resource occupancy、wave32/64、cache hint、solution mapping、active device、ranking determinism、gfx942/gfx950回帰、gfx1151 performance canaryを検証する。

## 5. Benchmark corpus

| category | range/example |
|---|---|
| Square | 16〜8192、2冪と中間値 |
| Tile boundary | MT境界±1/8/16/32 |
| Skinny-M/N | 1〜128 × large dimensions |
| Large-K | 少数output tile、K大 |
| Batched | batch 2〜数百 |
| Irregular | 非2冪、prime近傍、tail多数 |
| Product traces | 実ComfyUI/Anima trace |

FP16/BF16、NN/NT/TN/TT、beta 0/非0、alignment、fused epilogueを分ける。構成比はproduct trace 40%、境界30%、log-uniform random 30%。初期4,000〜8,000問題、全合法候補、複数SKUで数十万〜100万measurementを想定する。

## 6. ファイル別変更

| file | change |
|---|---|
| `shared/origami/include/origami/hardware.hpp` | WGP/CU/SIMD、register、cache/transaction、device profile |
| `shared/origami/src/origami/hardware.cpp` | runtime query、active device、WGP/CU検証 |
| `shared/origami/include/origami/types.hpp` | resource usage、wave、active WG/CU |
| `shared/origami/src/origami/gemm.cpp` | WGP-aware bandwidth、cache line、実WGM |
| `shared/origami/src/origami/heuristics.cpp` | gfx1151 scoped residual heuristic |
| `shared/origami/src/origami/streamk.cpp` | gfx1151 threshold/reduction/workspace |
| Tensile occupancy/serialization | resource metadata plumbing |
| RocRoller selection | occupancy/device固定廃止、候補・後段規則統合 |
| tests | resource/ranking/regression/device/performance |

## 7. リスク

| risk | mitigation |
|---|---|
| DVFS/temperature/cTDP noise | clock/temperature記録、順序randomize、boost/steady分離 |
| CPU shared-DRAM contention | isolated/contended profile分離 |
| 8060S overfit | 8050S/8040Sまたは別memory構成をholdout |
| ROCm/LLVM drift | build manifest、weekly canary、profile version |
| candidate explosion | resource prune + two-stage ranking |
| model/launch mismatch | launch config記録、単一決定系 |
| CDNA regression | architecture-scoped defaults、gfx942/gfx950 CI |

## 8. 最初の10営業日

1〜2日目: 環境固定、HIP properties/rocISA caps、CU/WGP/LDS/L2/register照合、active device修正。

3日目: 全solutionのwave、WG、VGPR/SGPR/LDS、CU occupancy、MT/MI/DepthU、WGM、cache hint、Stream-Kを一覧化。

4〜5日目: 2,000〜5,000問題、全候補実測、Origami CSV、oracle/regret/Top-k、最悪100件分類。

6〜8日目: WMMA、active WG、LDS、L2/MALL/DRAM、cache hint、CPU競合、epilogue/launch microbench。

9日目: resource schemaとmodel変更確定。

10日目: instrumentation、resource metadata、occupancy semantics、baseline reportの第1PR。

## 9. 停止条件

以下を満たさないまま係数fitへ進まない。

1. 実測CV > 2% → 環境制御を修正。
2. resource metadata欠損またはoccupancy=1固定 → plumbingを修正。
3. model WGMとlaunch WGM不一致 → integrationを修正。
4. current candidate oracle自体が遅い → candidate expansion優先。
5. shape-family holdout/別SKU悪化 → architectureとdevice/runtime parameterを再分離。

## 10. 工数と成果物

- 1名: 10〜12週間
- GPU性能2名 + data/measurement 0.5名: 6〜8暦週
- 総工数: 8〜12人週

成果物:

1. 再現可能なgfx1151 GEMM harness
2. 実測oracle dataset
3. architecture/device profile
4. resource-aware Origami schema
5. gfx1151係数/heuristic
6. expanded/pruned candidate generator
7. WGM/Stream-K統合
8. gfx942/gfx950 performance CI
9. regret/Top-k/outlier/SKU一般化report
10. READMEのgfx1151をOptimizedへ変更できる受入証跡

最重要順序は、**occupancyとwave/resource metadata修正 → 実測oracle → microbench → 構造係数fit → candidate expansion → residual heuristic → integration CI**である。現行定数だけを先に微調整すると、occupancy固定、wave欠損、後段Stream-K規則の誤差を別係数へ押し込むため採用しない。
