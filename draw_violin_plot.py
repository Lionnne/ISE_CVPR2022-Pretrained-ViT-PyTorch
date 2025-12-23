# comparison_plot.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import List, Optional, Dict

def load_evaluation_results(csv_files: List[str], model_names: Optional[List[str]] = None) -> List[dict]:
    """
    Load evaluation results from multiple CSV files.
    Returns a list of dicts, each containing 'model_name', 'per_class_acc' (list of 32 floats),
    'class_names' (list), 'class_avg_acc' (float).
    """
    results = []
    for i, csv_file in enumerate(csv_files):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # Assume consistent structure: class_name, class_id (0-31 for classes, -1 for average), per_class_acc, num_samples
        # Filter to only class rows (class_id >=0 and <32)
        class_df = df[df['class_id'] >= 0].sort_values('class_id')
        assert len(class_df) == 32, f"Expected 32 classes in {csv_file}, found {len(class_df)}"
        
        per_class_acc = class_df['per_class_acc'].tolist()
        class_names = class_df['class_name'].tolist()
        
        # Get class-averaged acc from the 'Class-Averaged' row
        avg_row = df[df['class_name'] == 'Class-Averaged']
        assert len(avg_row) == 1, f"Missing or duplicate Class-Averaged row in {csv_file}"
        class_avg_acc = avg_row['per_class_acc'].iloc[0]
        
        model_name = model_names[i] if model_names and i < len(model_names) else os.path.basename(csv_file).replace('.csv', '')
        results.append({
            'model_name': model_name,
            'per_class_acc': per_class_acc,
            'class_names': class_names,
            'class_avg_acc': class_avg_acc
        })
    
    # Ensure all have same class_names
    first_classes = results[0]['class_names']
    for res in results[1:]:
        assert res['class_names'] == first_classes, "Inconsistent class names across CSVs"
    
    return results

def load_abbrev_map(abbrev_file: str) -> Dict[str, str]:
    """
    Load class name to abbreviation mapping from CSV.
    """
    if not os.path.exists(abbrev_file):
        raise FileNotFoundError(f"Abbrev file not found: {abbrev_file}")
    
    df = pd.read_csv(abbrev_file)
    abbrev_map = dict(zip(df['class_name'], df['class_abbrev']))
    if len(abbrev_map) != 32:
        print(f"Warning: Expected 32 classes in abbrev file, found {len(abbrev_map)}")
    return abbrev_map

def create_abbrev(class_names: List[str], abbrev_map: Dict[str, str]) -> List[str]:
    """Create abbreviations using the provided mapping."""
    return [abbrev_map.get(name, name[:3]) for name in class_names]

def plot_comparison(results: List[dict], output_path: str, abbrev_map: Dict[str, str]):
    """Plot scatter of per-class acc for each model, with horizontal mean lines."""
    num_models = len(results)
    if num_models == 0:
        raise ValueError("No results to plot")
    
    class_names = results[0]['class_names']
    abbrev = create_abbrev(class_names, abbrev_map)
    class_ids = list(range(32))
    
    # Colors and markers for models
    colors = plt.cm.tab10.colors[:num_models]  # Use tab10 colormap
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:num_models]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for i, res in enumerate(results):
        model_name = res['model_name']
        per_class_acc = res['per_class_acc']
        class_avg = res['class_avg_acc']
        
        # Scatter plot for per-class acc
        ax.scatter(class_ids, per_class_acc, color=colors[i], marker=markers[i], s=50, 
                   label=f"{model_name} per-class", alpha=0.7)
        
        # Horizontal line for mean
        ax.axhline(y=class_avg, color=colors[i], linestyle='--', linewidth=2, 
                   label=f"{model_name} mean: {class_avg:.2f}%")
    
    # Formatting
    ax.set_xlabel('Class ID', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy Comparison Across Models', fontsize=14)
    ax.set_xticks(class_ids)
    ax.set_xticklabels(abbrev, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(0, 101, 10))
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_path}")

def plot_violin_distribution(results: List[dict], violin_output_path: str):
    """Plot violin distribution of per-class accuracies for each model."""
    num_models = len(results)
    if num_models == 0:
        raise ValueError("No results to plot")
    
    # Prepare data: list of lists of per_class_acc
    violin_data = [res['per_class_acc'] for res in results]
    model_names = [res['model_name'] for res in results]
    
    # Colors
    colors = plt.cm.tab10.colors[:num_models]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create violin plot without any internal lines; we'll add custom ones
    parts = ax.violinplot(violin_data, positions=range(num_models), showmeans=False, showmedians=False, showextrema=False)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Manually add min and max ticks (short horizontal lines, colored per model)
    for i in range(num_models):
        data = violin_data[i]
        min_val = min(data)
        max_val = max(data)
        # Min tick (at bottom)
        ax.hlines(y=min_val, xmin=i - 0.2, xmax=i + 0.2, color=colors[i], linewidth=2)
        # Max tick (at top)
        ax.hlines(y=max_val, xmin=i - 0.2, xmax=i + 0.2, color=colors[i], linewidth=2)
    
    # Add horizontal lines for mean and mean ± std inside each violin, with labels next to them
    for i, res in enumerate(results):
        mean_val = res['class_avg_acc']
        std_val = np.std(res['per_class_acc'])
        
        # Line for mean (thick solid)
        ax.hlines(y=mean_val, xmin=i - 0.25, xmax=i + 0.25, color=colors[i], linewidth=3, linestyle='-')
        # Label next to mean line (slightly to the right, at the line level)
        ax.text(i + 0.3, mean_val, f'μ={mean_val:.1f}%', ha='left', va='center', fontsize=8, 
                color=colors[i], fontweight='bold')
        
        # Lines for mean ± std (thin dashed, clipped to [0,100])
        # upper = min(100, mean_val + std_val)
        lower = max(0, mean_val - std_val)
        # ax.hlines(y=upper, xmin=i - 0.25, xmax=i + 0.25, color=colors[i], linewidth=1, linestyle='--', alpha=0.8)
        ax.hlines(y=lower, xmin=i - 0.25, xmax=i + 0.25, color=colors[i], linewidth=1, linestyle='--', alpha=0.8)
        # # Label next to upper dashed line
        # ax.text(i + 0.3, upper, f'+σ={std_val:.1f}%', ha='left', va='center', fontsize=8, 
        #         color=colors[i], alpha=0.8)
        # Label next to lower dashed line (same std value)
        ax.text(i + 0.3, lower, f'-σ={std_val:.1f}%', ha='left', va='center', fontsize=8, 
                color=colors[i], alpha=0.8)
    
    # Formatting
    ax.set_xticks(range(num_models))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Distribution of Per-Class Accuracies Across Models', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(violin_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Violin distribution plot saved to {violin_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare per-class accuracies across multiple model evaluations")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the comparison plot (e.g., /path/to/comparison.png)")
    parser.add_argument("--violin_output_path", type=str, required=True, help="Path to save the violin distribution plot (e.g., /path/to/violin_distribution.png)")
    parser.add_argument("--csv_files", type=str, nargs='+', required=True, 
                        help="Paths to evaluation_results.csv files (one per model)")
    parser.add_argument("--model_names", type=str, nargs='*', default=None, 
                        help="Optional: Names for models (if not provided, uses CSV filenames)")
    parser.add_argument("--abbrev_file", type=str, default="SO32_class_abbrev.csv", 
                        help="Path to class abbreviation CSV file (class_name,class_abbrev)")
    args = parser.parse_args()
    
    # Ensure model_names length matches if provided
    if args.model_names and len(args.model_names) != len(args.csv_files):
        raise ValueError("Number of model_names must match number of csv_files")
    
    results = load_evaluation_results(args.csv_files, args.model_names)
    abbrev_map = load_abbrev_map(args.abbrev_file)
    plot_comparison(results, args.output_path, abbrev_map)
    plot_violin_distribution(results, args.violin_output_path)
    
    # Optional: Print summary table
    print("\n=== Summary Table ===")
    summary_data = []
    for res in results:
        summary_data.append([res['model_name'], f"{res['class_avg_acc']:.2f}%"])
    summary_df = pd.DataFrame(summary_data, columns=['Model', 'Class-Averaged Acc'])
    print(summary_df.to_string(index=False))