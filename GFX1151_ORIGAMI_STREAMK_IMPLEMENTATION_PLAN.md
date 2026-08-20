# gfx1151 Origami / Stream-K 最適化方針

## 目的

Anima INT8 ConvRot の非量子化 GEMM と再構成 GEMMについて、Origami の候補削減が gfx1151 上の実測最速 Triton 構成を残すように校正する。Stream-K は独立に有効性を測り、通常 GEMM より速い形状にだけ適用する。

## 現状

- ROCm `develop` の Origami は gfx1151 を Functional として扱うが、Optimized の表示は gfx942/gfx950 のみ。
- gfx1151 の基本ハードウェア定数と RDNA WGP→物理 CU 補正は既に存在する。
- gfx942/gfx950 にはランキング回帰データや専用 heuristic がある一方、gfx1151 には同等の実測校正がない。
- 現在の Windows gfx1151 hipBLASLt では Stream-K 選択を要求しても通常解へフォールバックするため、環境変数だけでは高速化を確認できない。

## 実装手順

1. ROCm `develop` の Origami を Windows ROCm SDK でビルドできるようにする。これはビルド統合だけの変更とし、予測モデル変更と分離する。
2. Anima の主要 M/N/K、BF16/FP16、転置条件について Triton 候補を同一プロセス・GPU event・十分なウォームアップで測定する。
3. Origami の予測順位と実測順位を保存し、top-k recall、最速候補の取りこぼし、選択後レイテンシを比較する。
4. gfx1151 だけに限定した heuristic またはハードウェア係数を追加する。gfx942 の値をコピーせず、wave32、40 WGP / 80 CU、共有メモリ帯域、L2/MALL、行列命令の実測差を反映する。
5. 通常 GEMM、Origami、Stream-K、Origami + Stream-K を同じ形状・dtype・クロック状態で比較する。Stream-K が実際に選択されたかログでも確認する。
6. PyTorch Inductor の Origami top-k 経路で A/B 測定し、ComfyUI の30-step定常実行で最終確認する。

## 採用基準

- gfx1151 の対象形状集合で Origami top-8 が実測最速候補を安定して保持する。
- 選択後の中央値が Origami 無効時より悪化せず、コンパイル探索時間を短縮する。
- Stream-K は通常解より有意に速く、かつ実際に Stream-K kernel が選択された場合のみランチャー既定値にする。
- gfx942/gfx950および非ROCm経路の順位・動作を変更しない。

## ロールバック

Origami と Stream-K は別々の環境変数で無効化可能にする。実測で改善しない最適化枝はランチャー既定値へ入れない。
