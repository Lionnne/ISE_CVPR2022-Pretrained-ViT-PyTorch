# bar_plot_seeds.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict

def load_seed_results_from_csvs(seed_csv_files: List[str], dataset_groups: List[str], num_seeds: int = 5) -> Dict[str, List[float]]:
    """
    Load Class-Averaged acc from a list of CSV files, grouped by datasets.
    Assumes: len(seed_csv_files) == len(dataset_groups) == num_datasets * num_seeds
    dataset_groups: e.g., ['exfractal', 'exfractal', ..., 'imagenet', ...] (repeat per group)
    Returns: dict: dataset -> list of acc values (one per seed)
    """
    if len(seed_csv_files) != len(dataset_groups):
        raise ValueError("seed_csv_files and dataset_groups must have same length")
    
    num_datasets = len(set(dataset_groups))
    if len(seed_csv_files) != num_datasets * num_seeds:
        print(f"Warning: Expected {num_datasets * num_seeds} files, got {len(seed_csv_files)}")
    
    results = {}
    for dataset in set(dataset_groups):
        dataset_files = [f for i, f in enumerate(seed_csv_files) if dataset_groups[i] == dataset]
        accs = []
        for csv_file in dataset_files:
            if not os.path.exists(csv_file):
                print(f"Warning: CSV not found: {csv_file}. Skipping.")
                continue
            df = pd.read_csv(csv_file)
            avg_row = df[df['class_name'] == 'Class-Averaged']
            if len(avg_row) == 0:
                print(f"Warning: No Class-Averaged row in {csv_file}. Skipping.")
                continue
            class_avg_acc = avg_row['per_class_acc'].iloc[0]
            accs.append(class_avg_acc)
        results[dataset] = accs
        if len(accs) != num_seeds:
            print(f"Warning: Expected {num_seeds} seeds for {dataset}, found {len(accs)}")
    
    return results

def plot_bar_seeds(seed_results: Dict[str, List[float]], output_path: str):
    """Plot bar chart of mean Class-Averaged acc across seeds for each dataset, with std error bars."""
    datasets = list(seed_results.keys())
    means = [np.mean(accs) for accs in seed_results.values()]
    stds = [np.std(accs) for accs in seed_results.values()]
    
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
    ax.set_ylabel('Class-Averaged Accuracy (%)', fontsize=12)
    ax.set_title('Mean Class-Averaged Accuracy Across Random Seeds (with Std)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylim(75, max(max(means + [0]) + max(stds + [0]*5),90))  # Add some headroom
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar plot saved to {output_path}")

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
        "rcdb_SO32_repro114514/evaluation_results.csv"
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
        "rcdb", "rcdb", "rcdb", "rcdb", "rcdb"
    ]
    
    num_seeds = 5  # Seeds per dataset
    output_path = os.path.join(base_path,"bar_seeds.png")
    
    # Load results
    seed_results = load_seed_results_from_csvs(seed_csv_files, dataset_groups, num_seeds)
    
    # Plot
    plot_bar_seeds(seed_results, output_path)
    
    # Print summary table
    print("\n=== Summary Table (Seeds) ===")
    summary_data = []
    for dataset, accs in seed_results.items():
        if accs:  # Only if has data
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            summary_data.append([dataset, f"{mean_acc:.2f} ± {std_acc:.2f}%"])
    if summary_data:
        import pandas as pd  # Import here if not already
        summary_df = pd.DataFrame(summary_data, columns=['Dataset', 'Mean ± Std Acc'])
        print(summary_df.to_string(index=False))