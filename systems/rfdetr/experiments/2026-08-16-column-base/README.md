# 柱脚 2026-08-16 実験スクリプト

一日分の実験コード。**推論側で三件の改善、訓練側で四件の否定的結果**、
およびそれらを判定可能にした測定基盤。結論は
`docs/development_records/2026-08-16-column-base-measurement-floor.md`。

## 注意:絶対パスを含む

このセッションで動かした形のまま置いてある。`/workspace/exp_cb/`、
`/workspace/handoff_*/` などの絶対パスが埋め込まれているので、
**別環境ではパス定数の書き換えが必要**。整形して汎用化するより、
実際に数字を出した版をそのまま残すことを優先した。

## 既存スクリプトとの重複(重要)

`scripts/build_rfdetr_new_class_cv_folds.py` が既に存在し、そちらは
`scene_group_id` で折を切るため、同一シーンの近接ビューが折をまたがない。
本セッションの `make_cv_folds.py` はその保護を持たず、画像単位でしか
重複を防いでいない。

事後に pHash で近接群を再構成して確認した結果:224 枚中の近接群は 2 群
(4 枚、1.8%)、**両方とも折をまたいでいた**。影響は小さく、かつ全アーム
で等量なので対比較の結論は変わらないが、**新規に折を切る場合は既存の
`build_rfdetr_new_class_cv_folds.py` を使うこと**。

`make_bal_folds.py`(稀少グレードのオーバーサンプリング)も
`scripts/build_rfdetr_oversampled_view.py` と機能が重なる。

## 推論側(訓練なし、シード分散の影響を受けない)

| script | 役割 |
|---|---|
| `verify_column_base_freeze.py` | 納品構成の全数値を再計算(凍結パッケージの検証器) |
| `tta_fusion.py` | ビュー集合 × wbf_iou の探索(96 構成) |
| `view_search.py` | 実現可能率を目的とした視点・融合パラメータ探索 |
| `router_gate.py` | ルーターによる空間ゲート(部材外の箱を破棄) |
| `paired_fixed.py` / `paired_fusion_test.py` | 閾値固定の対ブートストラップ |
| `confirm_hflip.py` / `holdout_bright.py` | ホールドアウト検証(選択バイアスの除去) |
| `feasibility_test.py` | 四項達成の実現可能率、支配関係の検定 |
| `replicate_brace.py` | ブレース独立データでの再現確認 |
| `reoptimize_operating_point.py` | 検査員コストを目的とした動作点の再選択 |

## 測定基盤(交差検証)

| script | 役割 |
|---|---|
| `make_cv_folds.py` | 五折の構築(**上記の重複注意を参照**) |
| `cv_dump.py` | 折ごと・エポックごとの生検出の書き出し |
| `cv_aggregate.py` | 折を合算してから閾値探索(折内採点では小標本ノイズが残る) |
| `cv_falsealarm.py` / `cv_fa_aggregate.py` | 留保した健全画像での誤報 |
| `cv_feasibility.py` | 実現可能率(320 箱では飽和することの確認) |
| `run_cv_arm*.sh` | アーム実行(データセット / freeze-encoder / 解像度 / 短縮スケジュール) |

## 訓練側アームのデータ構築

| script | 役割 |
|---|---|
| `make_cp_folds.py` | copy-paste 合成陽性(折ごとに生成、来歴は構造的に清潔) |
| `make_cp2_folds.py` | 同上、貼付前に対象を切片の画素尺度へ縮小 |
| `make_bal_folds.py` | 稀少グレードのオーバーサンプリング |

## 診断

| script | 役割 |
|---|---|
| `fp_breakdown.py` | 誤報の内訳(位置誤り / グレード誤り / 重複) |
| `leakage_audit.py` | 訓練・試験間の近接重複監査(pHash + 画素余弦) |
| `inspector_cost.py` | 損傷側と誤報側を同一単位(検査員が見る箱数)へ換算 |
| `calibrate_router.py` | ルーターのクラス番号を既知データで実測較正 |
| `pool_presence.py` / `split_pool.py` / `deploy_profile.py` | 無注釈プールの中身確認(結果は撤回済み、経緯は開発記録参照) |
