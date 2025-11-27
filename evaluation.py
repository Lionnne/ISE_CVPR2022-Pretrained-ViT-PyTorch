# evaluate.py
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import argparse  # For command-line argument parsing
from typing import Dict, List

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

def create_abbrev(unique_classes: List[str], abbrev_map: Dict[str, str]) -> List[str]:
    """Create abbreviations using the provided mapping."""
    return [abbrev_map.get(name, name[:3]) for name in unique_classes]

def main(base_path, csv_file, class_map_file, abbrev_file=None):
    # Validate inputs
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    if not os.path.exists(class_map_file):
        raise FileNotFoundError(f"Class map file not found: {class_map_file}")

    df = pd.read_csv(csv_file)

    #######################################################################################################
    # Read the class_map file (assumes plain text format, one class name per line)
    with open(class_map_file, 'r', encoding='utf-8') as f:
        unique_classes = [line.strip() for line in f.readlines() if line.strip()]  # Read lines, strip whitespace, filter empty

    # Create mapping from class name to index (index starts from 0 based on file order)
    class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
    num_classes = len(unique_classes)  # Determined from file, should be 32
    assert num_classes == 32, f"Expected 31 classes, found {num_classes}"

    # Extract true labels: from the first part of filename (class name)
    df['true_class'] = df['filename'].apply(lambda x: x.split('/')[0])

    # Convert to numeric labels
    df['true_label'] = df['true_class'].map(class_to_id)  # Map using class_map
    df['pred_label'] = df['index']  # Predicted label is already an index

    # Check label ranges (handle mapping failures as NaN)
    df['true_label'] = df['true_label'].fillna(-1).astype(int)  # Convert NaN to -1 to avoid errors
    assert df['true_label'].min() >= 0 and df['true_label'].max() < num_classes, "True labels out of range"
    assert df['pred_label'].min() >= 0 and df['pred_label'].max() < num_classes, "Pred labels out of range"

    # Optional: Check for mapping failures (if true_class not in class_map)
    missing_classes = set(df['true_class'].unique()) - set(class_to_id.keys())
    if missing_classes:
        print(f"Warning: The following classes are not in class_map: {missing_classes}")
    #######################################################################################################

    # Overall Top-1 Accuracy
    total_samples = len(df)
    correct = (df['true_label'] == df['pred_label']).sum()
    overall_acc = correct / total_samples * 100

    # per-class accuracy
    per_class_acc = {}
    per_class_support = {}
    for cls_id in range(num_classes):
        class_mask = df['true_label'] == cls_id
        support = class_mask.sum()
        per_class_support[cls_id] = support
        if support > 0:
            class_correct = (df['pred_label'][class_mask] == cls_id).sum()
            per_class_acc[cls_id] = (class_correct / support) * 100
        else:
            per_class_acc[cls_id] = 0.0

    # Class-Averaged Accuracy
    class_avg_acc = np.mean([acc for acc, sup in zip(per_class_acc.values(), per_class_support.values()) if sup > 0])

    # Report（precision, recall, f1 per class；macro avg即class-averaged recall≈acc）
    report = classification_report(df['true_label'], df['pred_label'], 
                                   target_names=unique_classes, output_dict=True)
    macro_recall = report['macro avg']['recall'] * 100

    # Confusion Matrix
    cm = confusion_matrix(df['true_label'], df['pred_label'])

    # Output results
    print(f"\n=== Evaluation Results ===")
    print(f"Total Samples: {total_samples}")
    print(f"Overall Top-1 Accuracy: {overall_acc:.2f}%")
    print(f"Class-Averaged Accuracy (mean per-class acc): {class_avg_acc:.2f}%")
    print(f"Class-Averaged Recall (from sklearn macro avg): {macro_recall:.2f}%")  # 验证一致性

    # per-class accuracy details
    per_class_df = pd.DataFrame({
        'class_name': unique_classes,
        'class_id': list(range(num_classes)),
        'per_class_acc': [per_class_acc[i] for i in range(num_classes)],
        'num_samples': [per_class_support[i] for i in range(num_classes)]
    })
    # Add Class-Averaged row at the end
    avg_row = pd.DataFrame({
        'class_name': ['Class-Averaged'],
        'class_id': [-1],  # Placeholder for average
        'per_class_acc': [class_avg_acc],
        'num_samples': [total_samples]  # Total samples for reference
    })
    per_class_df = pd.concat([per_class_df, avg_row], ignore_index=True)

    # save detailed results to CSV
    output_csv = os.path.join(base_path, 'evaluation_results.csv')
    per_class_df.to_csv(output_csv, index=False)

    # save confusion matrix to CSV
    cm_df = pd.DataFrame(cm, index=unique_classes, columns=unique_classes)
    cm_df.to_csv(os.path.join(base_path, 'confusion_matrix.csv'))

    # Plot and save confusion matrix as image
    # Load abbreviations from CSV if provided, else fallback to hardcoded
    if abbrev_file:
        abbrev_map = load_abbrev_map(abbrev_file)
        abbrev = create_abbrev(unique_classes, abbrev_map)
    else:
        abbrev = [name for name in unique_classes]

    # Normalize confusion matrix by row (each row sums to 1)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalize_cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)

    # Plot normalized confusion matrix using matplotlib (heatmap style)
    plt.figure(figsize=(16, 14))  # Adjust size for 32 classes
    im = plt.imshow(normalize_cm, cmap='Blues', interpolation='nearest', aspect='auto')
    plt.colorbar(im, shrink=1.0, aspect=50, pad=0.02)

    # Set labels
    plt.xticks(range(num_classes), abbrev, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(num_classes), abbrev, rotation=0, fontsize=8)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Normalized Confusion Matrix (rows sum to 1)', fontsize=14)

    # Add text annotations: show normalized values with 2 decimal places
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f'{normalize_cm[i, j]:.2f}', ha='center', va='center', 
                     color='black' if normalize_cm[i, j] < 0.5 else 'white', fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Normalized confusion matrix plot saved to {os.path.join(base_path, 'confusion_matrix.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions on SO32 dataset")
    parser.add_argument("--base_path", type=str, required=True, help="Base path for outputs")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to predictions CSV")
    parser.add_argument("--class_map_file", type=str, required=True, help="Path to class map TXT")
    parser.add_argument("--abbrev_file", type=str, default="SO32_class_abbrev.csv", 
                        help="Path to class abbreviation CSV file (class_name,class_abbrev); if not provided, uses fallback abbreviations")
    args = parser.parse_args()
    main(args.base_path, args.csv_file, args.class_map_file, args.abbrev_file)