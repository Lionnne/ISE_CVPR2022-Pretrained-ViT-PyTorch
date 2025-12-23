# bar_plot_seeds.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict

def load_seed_results_from_csvs(seed_csv_files: List[str], dataset_groups: List[str], num_seeds: int = 5) -> Dict[str, Dict[str, List[float]]]:
    """
    Load both Class-Averaged and Overall Top-1 acc from a list of CSV files, grouped by datasets.
    Assumes: len(seed_csv_files) == len(dataset_groups) == num_datasets * num_seeds
    dataset_groups: e.g., ['exfractal', 'exfractal', ..., 'imagenet', ...] (repeat per group)
    Returns: dict: dataset -> {'class_avg': list of acc values, 'overall': list of acc values} (one per seed)
    """
    if len(seed_csv_files) != len(dataset_groups):
        raise ValueError("seed_csv_files and dataset_groups must have same length")
    
    num_datasets = len(set(dataset_groups))
    if len(seed_csv_files) != num_datasets * num_seeds:
        print(f"Warning: Expected {num_datasets * num_seeds} files, got {len(seed_csv_files)}")
    
    results = {}
    for dataset in set(dataset_groups):
        dataset_files = [f for i, f in enumerate(seed_csv_files) if dataset_groups[i] == dataset]
        class_avg_accs = []
        overall_accs = []
        for csv_file in dataset_files:
            if not os.path.exists(csv_file):
                print(f"Warning: CSV not found: {csv_file}. Skipping.")
                continue
            df = pd.read_csv(csv_file)
            # Load Class-Averaged
            avg_row = df[df['class_name'] == 'Class-Averaged']
            if len(avg_row) == 0:
                print(f"Warning: No Class-Averaged row in {csv_file}. Skipping.")
                continue
            class_avg_acc = avg_row['per_class_acc'].iloc[0]
            class_avg_accs.append(class_avg_acc)
            # Load Overall Top-1
            overall_row = df[df['class_name'] == 'Overall Top-1']
            if len(overall_row) == 0:
                print(f"Warning: No Overall Top-1 row in {csv_file}. Skipping.")
                continue
            overall_acc = overall_row['per_class_acc'].iloc[0]
            overall_accs.append(overall_acc)
        results[dataset] = {'class_avg': class_avg_accs, 'overall': overall_accs}
        if len(class_avg_accs) != num_seeds:
            print(f"Warning: Expected {num_seeds} seeds for {dataset}, found {len(class_avg_accs)}")
    
    return results

def plot_bar_seeds(seed_results: Dict[str, Dict[str, List[float]]], output_path: str, metric: str = 'class_avg'):
    """Plot bar chart of mean acc across seeds for each dataset, with std error bars."""
    fixed_order = ['imagenet', 'exfractal', 'rcdb', 'ConvNeXT','Swin Transformer']
    datasets = [ds for ds in fixed_order if ds in seed_results]
    accs_list = [seed_results[ds][metric] for ds in datasets]
    means = [np.mean(accs) for accs in accs_list]
    stds = [np.std(accs) for accs in accs_list]
    
    if len(datasets) == 0:
        raise ValueError("No results to plot")
    
    # Colors
    colors = plt.cm.tab10.colors[:len(datasets)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bar plot with error bars
    x_pos = np.arange(len(datasets))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on top of bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std*1.5,
                f'{mean:.2f} ± {std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()} Accuracy (%)', fontsize=12)
    ax.set_title(f'Mean {metric.replace("_", " ").title()} Accuracy Across Random Seeds (with Std)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylim(max(means+[0])-10, max(means + [0]) + max(max(stds + [0])*2,2))  # Add some headroom
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar plot for {metric} saved to {output_path}")

if __name__ == "__main__":
    # Hardcoded CSV paths: 3 datasets x 5 seeds = 15 files
    # Modify these paths to match your actual file locations
    seed_csv_files = [
        # Exfractal seeds (first 5)
        "exfractal_SO32_repro42/evaluation_results.csv",
        "exfractal_SO32_repro256/evaluation_results.csv",
        "exfractal_SO32_repro666/evaluation_results.csv",
        "exfractal_SO32_repro3407/evaluation_results.csv",
        "exfractal_SO32_repro114514/evaluation_results.csv",
        # Imagenet seeds (next 5)
        "imagenet_SO32_repro42/evaluation_results.csv",
        "imagenet_SO32_repro256/evaluation_results.csv",
        "imagenet_SO32_repro666/evaluation_results.csv",
        "imagenet_SO32_repro3407/evaluation_results.csv",
        "imagenet_SO32_repro114514/evaluation_results.csv",
        # RCDB seeds (last 5)
        "rcdb_SO32_repro42/evaluation_results.csv",
        "rcdb_SO32_repro256/evaluation_results.csv",
        "rcdb_SO32_repro666/evaluation_results.csv",
        "rcdb_SO32_repro3407/evaluation_results.csv",
        "rcdb_SO32_repro114514/evaluation_results.csv",
        # ConvNeXT seeds (last 5)
        "convnext_base.fb_in22k_ft_in1kSO32_42/evaluation_results.csv",
        "convnext_base.fb_in22k_ft_in1kSO32_256/evaluation_results.csv",
        "convnext_base.fb_in22k_ft_in1kSO32_666/evaluation_results.csv",
        "convnext_base.fb_in22k_ft_in1kSO32_3407/evaluation_results.csv",
        "convnext_base.fb_in22k_ft_in1kSO32_114514/evaluation_results.csv",
        # Swin Transformer
        "swin_base_patch4_window7_224.ms_in22k_SO32_1.0e-4_adamw_wd1e-4_aug42/evaluation_results.csv",
        "swin_base_patch4_window7_224.ms_in22k_SO32_1.0e-4_adamw_wd1e-4_aug256/evaluation_results.csv",
        "swin_base_patch4_window7_224.ms_in22k_SO32_1.0e-4_adamw_wd1e-4_aug666/evaluation_results.csv",
        "swin_base_patch4_window7_224.ms_in22k_SO32_1.0e-4_adamw_wd1e-4_aug3407/evaluation_results.csv",
        "swin_base_patch4_window7_224.ms_in22k_SO32_1.0e-4_adamw_wd1e-4_aug114514/evaluation_results.csv"
    ]
    base_path = "./inference_results"  # Base path to prepend if needed
    seed_csv_files = [os.path.join(base_path, f) for f in seed_csv_files]
    
    # Corresponding dataset groups (repeat name 5 times per dataset)
    dataset_groups = [
        # Exfractal
        "exfractal", "exfractal", "exfractal", "exfractal", "exfractal",
        # Imagenet
        "imagenet", "imagenet", "imagenet", "imagenet", "imagenet",
        # RCDB
        "rcdb", "rcdb", "rcdb", "rcdb", "rcdb",
        # ConvNeXT
        "ConvNeXT","ConvNeXT","ConvNeXT","ConvNeXT","ConvNeXT",
        # Swin Transformer
        "Swin Transformer","Swin Transformer","Swin Transformer","Swin Transformer","Swin Transformer",
    ]
    
    num_seeds = 5  # Seeds per dataset
    output_path_class = os.path.join(base_path, "bar_class_avg_2.png")
    output_path_overall = os.path.join(base_path, "bar_overall_2.png")
    
    # Load results
    seed_results = load_seed_results_from_csvs(seed_csv_files, dataset_groups, num_seeds)
    
    # Plot for Class-Averaged
    plot_bar_seeds(seed_results, output_path_class, 'class_avg')
    
    # Plot for Overall Top-1
    plot_bar_seeds(seed_results, output_path_overall, 'overall')
    
    # Print summary table
    print("\n=== Summary Table (Seeds) ===")
    fixed_order = ['imagenet', 'exfractal', 'rcdb','ConvNeXT','Swin Transformer']
    summary_data = []
    for dataset in fixed_order:
        if dataset in seed_results:
            class_accs = seed_results[dataset]['class_avg']
            overall_accs = seed_results[dataset]['overall']
            if class_accs and overall_accs:  # Only if has data
                class_mean = np.mean(class_accs)
                class_std = np.std(class_accs)
                overall_mean = np.mean(overall_accs)
                overall_std = np.std(overall_accs)
                summary_data.append([dataset, f"{class_mean:.2f} ± {class_std:.2f}%", f"{overall_mean:.2f} ± {overall_std:.2f}%"])
    if summary_data:
        summary_df = pd.DataFrame(summary_data, columns=['Dataset', 'Class-Avg Mean ± Std', 'Overall Top-1 Mean ± Std'])
        print(summary_df.to_string(index=False))