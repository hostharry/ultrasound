"""
Adaptive-HASA 与 TV-only 方法在肿瘤ROI区域SSIM的统计比较
============================================================

本脚本验证假设：Adaptive-HASA 在肿瘤ROI区域的SSIM显著高于TV-only方法

统计方法：单侧配对 Wilcoxon 签名秩检验
- H0: Adaptive-HASA SSIM = TV-only SSIM
- H1: Adaptive-HASA SSIM > TV-only SSIM
- 显著性水平: α = 0.05
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CONFIG = {
    'data_file': 'reconstruction_all_methods.csv',
    'alpha': 0.05,  # 显著性水平
    'method_baseline': 'TV-only',
    'method_test': 'Adaptive-HASA',
    'metric': 'SSIM_Tumor',
}


# ==================== 数据加载与处理 ====================

def load_data(filepath):
    """加载重建结果数据集"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到数据文件: {filepath}")
    
    data = pd.read_csv(filepath)
    print(f"数据集形状: {data.shape}")
    print(f"列名: {data.columns.tolist()}")
    print(f"\n方法列表: {data['Method'].unique()}")
    return data


def extract_method_data(data, method_name):
    """提取指定方法的数据，按图像名称排序"""
    method_data = data[data['Method'] == method_name].copy()
    method_data = method_data.sort_values('Image').reset_index(drop=True)
    return method_data


def verify_alignment(data1, data2, col='Image'):
    """验证两组数据的图像对齐"""
    aligned = np.all(data1[col].values == data2[col].values)
    if not aligned:
        raise ValueError("两组数据的图像未对齐！")
    return True


# ==================== 统计检验 ====================

def perform_wilcoxon_test(ssim_test, ssim_baseline, alternative='greater'):
    """
    执行单侧配对 Wilcoxon 签名秩检验
    
    参数:
        ssim_test: 测试方法的SSIM值
        ssim_baseline: 基准方法的SSIM值
        alternative: 'greater' 表示检验 test > baseline
    
    返回:
        stat: 检验统计量
        pvalue: p值
    """
    stat, pvalue = wilcoxon(ssim_test, ssim_baseline, alternative=alternative)
    return stat, pvalue


def compute_descriptive_stats(ssim_values, name):
    """计算描述性统计"""
    return {
        'name': name,
        'mean': np.mean(ssim_values),
        'std': np.std(ssim_values),
        'median': np.median(ssim_values),
        'min': np.min(ssim_values),
        'max': np.max(ssim_values),
    }


# ==================== 结果展示 ====================

def print_descriptive_stats(stats):
    """打印描述性统计"""
    print(f"  {stats['name']}:")
    print(f"    均值: {stats['mean']:.6f}")
    print(f"    标准差: {stats['std']:.6f}")
    print(f"    中位数: {stats['median']:.6f}")
    print(f"    范围: [{stats['min']:.6f}, {stats['max']:.6f}]")


def create_comparison_table(baseline_data, test_data, ssim_baseline, ssim_test):
    """创建配对比较表"""
    differences = ssim_test - ssim_baseline
    
    comparison_df = pd.DataFrame({
        '图像': baseline_data['Image'].values,
        '类型': baseline_data['Type'].values,
        f'{CONFIG["method_baseline"]} SSIM': ssim_baseline,
        f'{CONFIG["method_test"]} SSIM': ssim_test,
        '差值': differences,
        'Adaptive更优': differences > 0
    })
    
    return comparison_df, differences


def analyze_by_tumor_type(comparison_df):
    """按肿瘤类型分析结果"""
    results = {}
    
    for tumor_type in ['benign', 'malignant']:
        mask = comparison_df['类型'] == tumor_type
        subset = comparison_df[mask]
        
        results[tumor_type] = {
            'count': mask.sum(),
            'baseline_mean': subset[f'{CONFIG["method_baseline"]} SSIM'].mean(),
            'test_mean': subset[f'{CONFIG["method_test"]} SSIM'].mean(),
            'diff_mean': subset['差值'].mean(),
            'wins': subset['Adaptive更优'].sum(),
        }
    
    return results


def print_tumor_type_analysis(type_results):
    """打印按肿瘤类型的分析结果"""
    print("\n按肿瘤类型分析:")
    
    for tumor_type, stats in type_results.items():
        type_name = "良性" if tumor_type == "benign" else "恶性"
        print(f"\n  {type_name}肿瘤 (n={stats['count']}):")
        print(f"    {CONFIG['method_baseline']} 平均SSIM: {stats['baseline_mean']:.6f}")
        print(f"    {CONFIG['method_test']} 平均SSIM: {stats['test_mean']:.6f}")
        print(f"    平均差值: {stats['diff_mean']:.6f}")
        print(f"    Adaptive更优: {stats['wins']}/{stats['count']} 例")


def create_visualization(ssim_baseline, ssim_test, differences, output_path='tumor_roi_ssim_comparison.png'):
    """创建可视化图表"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 箱线图比较
    ax1 = axes[0]
    box_data = [ssim_baseline, ssim_test]
    bp = ax1.boxplot(box_data, labels=[CONFIG['method_baseline'], CONFIG['method_test']], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('Tumor ROI SSIM')
    ax1.set_title('方法比较 - 箱线图')
    ax1.grid(True, alpha=0.3)
    
    # 2. 配对散点图
    ax2 = axes[1]
    ax2.scatter(ssim_baseline, ssim_test, alpha=0.7, edgecolors='black', linewidth=0.5)
    min_val = min(ssim_baseline.min(), ssim_test.min()) - 0.02
    max_val = max(ssim_baseline.max(), ssim_test.max()) + 0.02
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (无差异线)')
    ax2.set_xlabel(f'{CONFIG["method_baseline"]} SSIM')
    ax2.set_ylabel(f'{CONFIG["method_test"]} SSIM')
    ax2.set_title('配对散点图')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    
    # 3. 差值直方图
    ax3 = axes[2]
    ax3.hist(differences, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='无差异')
    ax3.axvline(x=differences.mean(), color='green', linestyle='-', linewidth=2, label=f'平均差值={differences.mean():.4f}')
    ax3.set_xlabel('SSIM差值 (Adaptive-HASA - TV-only)')
    ax3.set_ylabel('频数')
    ax3.set_title('差值分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化图已保存: {output_path}")
    plt.show()


def print_final_conclusion(stat, pvalue, differences, alpha):
    """打印最终结论"""
    wins = np.sum(differences > 0)
    losses = np.sum(differences < 0)
    ties = np.sum(differences == 0)
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    if pvalue < alpha:
        print(f"\n✓ 假设得到支持!")
        print(f"\n统计检验结果显示，Adaptive-HASA 在肿瘤ROI区域的SSIM")
        print(f"显著高于 TV-only 方法 (p = {pvalue:.2e} < {alpha})")
    else:
        print(f"\n✗ 假设未得到支持")
        print(f"\n统计检验结果未显示显著差异 (p = {pvalue:.4f} ≥ {alpha})")
    
    print(f"\n详细统计:")
    print(f"  - Adaptive-HASA 更优: {wins}/{len(differences)} 例 ({wins/len(differences)*100:.1f}%)")
    print(f"  - TV-only 更优: {losses}/{len(differences)} 例 ({losses/len(differences)*100:.1f}%)")
    print(f"  - 相同: {ties}/{len(differences)} 例")
    print(f"  - 平均SSIM提升: {differences.mean():.6f}")
    print(f"  - 最大SSIM提升: {differences.max():.6f}")
    
    print("=" * 70)


# ==================== 主函数 ====================

def main():
    print("=" * 70)
    print("Adaptive-HASA vs TV-only 肿瘤ROI区域SSIM统计分析")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    data = load_data(CONFIG['data_file'])
    
    # 2. 提取两种方法的数据
    print(f"\n[2] 提取方法数据...")
    baseline_data = extract_method_data(data, CONFIG['method_baseline'])
    test_data = extract_method_data(data, CONFIG['method_test'])
    
    print(f"  {CONFIG['method_baseline']} 样本数: {len(baseline_data)}")
    print(f"  {CONFIG['method_test']} 样本数: {len(test_data)}")
    
    # 3. 验证数据对齐
    verify_alignment(baseline_data, test_data)
    print("  ✓ 数据对齐验证通过")
    
    # 4. 提取SSIM值
    ssim_baseline = baseline_data[CONFIG['metric']].values
    ssim_test = test_data[CONFIG['metric']].values
    
    # 5. 描述性统计
    print(f"\n[3] 描述性统计 ({CONFIG['metric']}):")
    stats_baseline = compute_descriptive_stats(ssim_baseline, CONFIG['method_baseline'])
    stats_test = compute_descriptive_stats(ssim_test, CONFIG['method_test'])
    print_descriptive_stats(stats_baseline)
    print_descriptive_stats(stats_test)
    
    # 6. Wilcoxon 检验
    print(f"\n[4] Wilcoxon 签名秩检验:")
    stat, pvalue = perform_wilcoxon_test(ssim_test, ssim_baseline, alternative='greater')
    
    print(f"  H0: {CONFIG['method_test']} SSIM = {CONFIG['method_baseline']} SSIM")
    print(f"  H1: {CONFIG['method_test']} SSIM > {CONFIG['method_baseline']} SSIM")
    print(f"  检验统计量 (W+): {stat}")
    print(f"  p值: {pvalue:.2e}")
    print(f"  显著性水平 α: {CONFIG['alpha']}")
    print(f"  结果: {'显著 (拒绝H0)' if pvalue < CONFIG['alpha'] else '不显著 (无法拒绝H0)'}")
    
    # 7. 创建比较表
    print(f"\n[5] 配对比较详情:")
    comparison_df, differences = create_comparison_table(
        baseline_data, test_data, ssim_baseline, ssim_test
    )
    print(comparison_df.to_string(index=False))
    
    # 8. 按肿瘤类型分析
    type_results = analyze_by_tumor_type(comparison_df)
    print_tumor_type_analysis(type_results)
    
    # 9. 最终结论
    print_final_conclusion(stat, pvalue, differences, CONFIG['alpha'])
    
    # 10. 可视化
    print("\n[6] 生成可视化...")
    create_visualization(ssim_baseline, ssim_test, differences)
    
    return {
        'stat': stat,
        'pvalue': pvalue,
        'comparison_df': comparison_df,
        'differences': differences,
        'type_results': type_results,
    }


if __name__ == '__main__':
    results = main()
