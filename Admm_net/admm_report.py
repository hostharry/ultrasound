import json
import os
from datetime import datetime
from typing import Dict, List

from admm_metrics import summarize_metrics


def generate_evaluation_report(metrics_list: List[Dict[str, float]], meta: dict, save_dir: str, history: Dict[str, List[float]]) -> str:
    args_dict = meta.get("args", {})
    agg = summarize_metrics(metrics_list)
    lines = [
        "# HASA-ADMM-Net 评估报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 模型配置",
        "",
        f"- 展开层数: {args_dict.get('layers', 'N/A')}",
        f"- W_mode: {args_dict.get('W_mode', 'N/A')}",
        f"- 压缩比: {args_dict.get('cs_ratio', 'N/A')}",
        "",
        "## 2. 指标均值 ± 标准差",
        "",
        f"- SNR: {agg.get('SNR_dB_mean', 0):.2f} ± {agg.get('SNR_dB_std', 0):.2f} dB",
        f"- NMSE: {agg.get('NMSE_mean', 0):.6f} ± {agg.get('NMSE_std', 0):.6f}",
        f"- PSNR: {agg.get('PSNR_dB_mean', 0):.2f} ± {agg.get('PSNR_dB_std', 0):.2f} dB",
        f"- SSIM(1D): {agg.get('SSIM_1D_mean', 0):.4f} ± {agg.get('SSIM_1D_std', 0):.4f}",
        f"- Env Corr: {agg.get('Env_Corr_mean', 0):.4f} ± {agg.get('Env_Corr_std', 0):.4f}",
        "",
    ]
    if history.get("epoch"):
        lines += [
            "## 3. 训练日志摘要",
            "",
            f"- 总 Epoch 数: {history['epoch'][-1]}",
            f"- 最终 Val SNR: {history['val_snr'][-1]:.2f} dB",
            f"- 最终 Val NMSE: {history['val_nmse'][-1]:.6f}",
            "",
        ]
    report_path = os.path.join(save_dir, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def save_eval_summary(save_dir: str, summary: dict):
    out_path = os.path.join(save_dir, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return out_path
