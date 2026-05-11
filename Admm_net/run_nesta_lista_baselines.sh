#!/usr/bin/env bash
set -euo pipefail

# 统一运行 Classical NESTA（评估）、Deep-Unfolded NESTA（训练）、LISTA（训练）
# 支持两个数据源：PICMUS simu_reso 与 FDBF npz。
# 用法示例：
#   bash run_nesta_lista_baselines.sh                 # 两个数据集都跑
#   bash run_nesta_lista_baselines.sh picmus          # 只跑 PICMUS
#   bash run_nesta_lista_baselines.sh fdbf            # 只跑 FDBF
#   GPU=1 bash run_nesta_lista_baselines.sh picmus

MODE="${1:-all}"    # all | picmus | fdbf
GPU="${GPU:-0}"
CS_RATIO="${CS_RATIO:-8}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-200}"

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

run_picmus() {
  echo "========================================"
  echo "[PICMUS] Classical NESTA / NESTA / LISTA"
  echo "========================================"

  python evaluate_NESTA_Classical.py \
    --npz ../data/picmus_simu_reso.npz \
    --cs_ratio "$CS_RATIO" \
    --n_iters 60 \
    --n_restarts 0 \
    --eta 1e-4 \
    --transform identity \
    --eval_all \
    --gpu "$GPU"

  python train_NESTA.py \
    --npz ../data/picmus_simu_reso.npz \
    --cs_ratio "$CS_RATIO" \
    --val_ratio 0.1 \
    --split_mode group \
    --seed "$SEED" \
    --layers 15 \
    --gamma_env 0.1 \
    --loss_mode nmse \
    --epochs "$EPOCHS" \
    --batch_size 16 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --warm_restarts 40 \
    --grad_clip 1.0 \
    --gpu "$GPU"

  python train_LISTA.py \
    --npz ../data/picmus_simu_reso.npz \
    --cs_ratio "$CS_RATIO" \
    --val_ratio 0.1 \
    --split_mode group \
    --seed "$SEED" \
    --layers 30 \
    --kernel_size 5 \
    --gamma_env 0.1 \
    --loss_mode nmse \
    --epochs "$EPOCHS" \
    --batch_size 4 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --warm_restarts 40 \
    --grad_clip 1.0 \
    --gpu "$GPU"
}

run_fdbf() {
  echo "======================================"
  echo "[FDBF] Classical NESTA / NESTA / LISTA"
  echo "======================================"

  python evaluate_NESTA_Classical.py \
    --npz ../dataset_fdbf_energy_mu_8_9_15.npz \
    --cs_ratio "$CS_RATIO" \
    --n_iters 60 \
    --n_restarts 0 \
    --eta 1e-4 \
    --transform identity \
    --eval_all \
    --gpu "$GPU"

  python train_NESTA.py \
    --npz ../dataset_fdbf_energy_mu_8_9_15.npz \
    --cs_ratio "$CS_RATIO" \
    --val_ratio 0.1 \
    --split_mode random \
    --seed "$SEED" \
    --layers 15 \
    --gamma_env 0.1 \
    --loss_mode nmse \
    --epochs "$EPOCHS" \
    --batch_size 16 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --warm_restarts 40 \
    --grad_clip 1.0 \
    --gpu "$GPU"

  python train_LISTA.py \
    --npz ../dataset_fdbf_energy_mu_8_9_15.npz \
    --cs_ratio "$CS_RATIO" \
    --val_ratio 0.1 \
    --split_mode random \
    --seed "$SEED" \
    --layers 30 \
    --kernel_size 5 \
    --gamma_env 0.1 \
    --loss_mode nmse \
    --epochs "$EPOCHS" \
    --batch_size 4 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --warm_restarts 40 \
    --grad_clip 1.0 \
    --gpu "$GPU"
}

case "$MODE" in
  all)
    run_picmus
    run_fdbf
    ;;
  picmus)
    run_picmus
    ;;
  fdbf)
    run_fdbf
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash run_nesta_lista_baselines.sh [all|picmus|fdbf]"
    exit 1
    ;;
esac

