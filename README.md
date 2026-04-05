# ESC-50 Pooling Protocol (Fixed Lightweight CNN)

## 고정 조건

- Dataset: ESC-50 official 5-fold CV
- Protocol: `4 folds train` + `1 fold eval` (fold별 hold-out test), 평균으로 최종 보고
- If `cv_protocol=fold_val`: validation is computed every epoch on the hold-out fold, and best val accuracy is averaged across folds.
- Input: log-mel `(431, 40, 1)` (5s @ 44.1kHz)
- Backbone: paper-style lightweight CNN (Conv 32/64/128, AvgPool after conv1/2)
- Optimizer: SGD (`lr=0.05`, `momentum=0.9`), `batch=64`, `epochs=700`
- Augmentation: Mixup `alpha=0.2`
- LR schedule: `ReduceLROnPlateau` on `val_accuracy` (`factor=0.5`, `patience=20`, `cooldown=5`, `min_lr=1e-4`)
- Adaptive regularization: entropy bonus (`loss = loss_task - lambda * H(alpha)`, default `lambda=1e-3`)
- Input: fixed log-mel spectrogram (`input_representation: mel`)
- Backbone: fixed `lightweight_cnn`
- Optimizer/LR schedule/batch size/epochs/seed: 설정값 고정
- Augmentation: 설정값 고정 (기본 템플릿은 OFF)
- 비교 변수: pooling만 변경

## 실험군

- P0: `GAP`
- P1: `GMP`
- P2: `SSRP-T (W=4, K=12)`
- P3: `SSRP-T (W=2, K=4)`
- P4: `AdaptiveSSRP-T (W=4, Ks=4/8/12, h=128)`
- P5: `AdaptiveSSRP-T (W=2, Ks=2/4/6, h=64)`

## 로그/저장 항목

각 fold/seed마다:
- best val accuracy (또는 설정된 best metric)
- test accuracy / macro-F1
- training curve (`history_YYYYMMDD_HHMMSS.json/.csv`, plus stable `history.json/.csv`)
- pooling 적용 직전 feature map 시간 길이 `T` (`pooling_feature_time_lengths`)
- Adaptive일 때 alpha 통계:
  - 전체 평균 `alpha_mean`
  - 클래스별 평균 `alpha_mean_by_class`
  - collapse 지표: `alpha_entropy_norm_mean`, `alpha_argmax_dominant_ratio`
  - 조건별(temporal short/long) alpha 평균: `temporal_conditioning.short_term/long_term`

## 실험명 규칙

`run_protocol.py`가 아래 형식으로 자동 생성:
- `esc50_lwcnn_pool_GAP_seed42`
- `esc50_lwcnn_pool_SSRPT_W4K12_seed42`
- `esc50_lwcnn_pool_AdaptSSRP_W2_Ks246_h64_seed43`

## 결과 파일

- `<out_dir>/<run_name>/cv_summary.json`
- `<out_dir>/<run_name>/cv_summary.csv`
- `<out_dir>/<run_name>/resolved_config_YYYYMMDD_HHMMSS.yaml` (+ stable `resolved_config.yaml`)
- `<out_dir>/<run_name>/run_params_YYYYMMDD_HHMMSS.json`
- `<out_dir>/<run_name>/<fold>/result_YYYYMMDD_HHMMSS.json` (+ stable `result.json`)
- `<out_dir>/<run_name>/<fold>/history_YYYYMMDD_HHMMSS.json` (+ stable `history.json`)
- `<out_dir>/<run_name>/<fold>/history_YYYYMMDD_HHMMSS.csv` (+ stable `history.csv`)
- `<out_dir>/<run_name>/<fold>/best_YYYYMMDD_HHMMSS.pt` (if `train.checkpoint_timestamp=true`)

프로토콜 종합:
- `<out_dir>/protocol_YYYYMMDD_HHMMSS.json`
- `<out_dir>/protocol_results_YYYYMMDD_HHMMSS.json`
- `<out_dir>/table1_pooling_comparison_YYYYMMDD_HHMMSS.csv`
- `<out_dir>/table1_pooling_comparison_YYYYMMDD_HHMMSS.md`
