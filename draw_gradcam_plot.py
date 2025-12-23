import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle


def make_gradcam_gallery():
    # Path to visualization results
    VIS_FOLDER = r"/ldap_shared/home/s_ljy/Projects/CVPR2022-Pretrained-ViT-PyTorch/inference_results/swin_base_patch4_window7_224.ms_in22k_SO32_1.0e-4_adamw_wd1e-4_aug42/vis_results_2"

    # Derive title from parent folder name
    parent_dir = os.path.dirname(VIS_FOLDER.rstrip("/\\"))
    folder_name = os.path.basename(parent_dir)
    title = folder_name.replace("_vis_results", "").strip("_")

    # Output path for the final gallery image
    output_path = os.path.join(VIS_FOLDER, f"{title}_gradcam32.png")

    random.seed(42)  # For reproducible random selection

    # Get all class subfolders and sort them
    class_dirs = sorted([d for d in os.listdir(VIS_FOLDER)
                         if os.path.isdir(os.path.join(VIS_FOLDER, d))])

    if len(class_dirs) != 32:
        print(f"Warning: Found {len(class_dirs)} classes, expected 32.")

    # Create 4×8 grid (large figure to accommodate long class names)
    fig, axes = plt.subplots(4, 8, figsize=(48, 20))
    axes = axes.flatten()

    for idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(VIS_FOLDER, class_name)
        img_files = [f for f in os.listdir(class_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        # If no image in this class
        if not img_files:
            axes[idx].axis('off')
            axes[idx].text(0.5, 0.5, class_name + "\n(No Image)",
                           ha='center', va='center', fontsize=12, color='red')
            continue

        # Randomly pick one image per class
        chosen = random.choice(img_files)
        img_path = os.path.join(class_path, chosen)
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')

        # Determine border/title color based on prediction correctness
        name_lower = chosen.lower()
        if "true" in name_lower:
            border_color = "#0066ff"   # Correct prediction → blue
            title_color = "black"
        elif "false" in name_lower:
            border_color = "#ff0000"   # Wrong prediction → red
            title_color = "red"
        else:
            border_color = "gray"      # Unknown → gray
            title_color = "black"

        # Thick colored border around each subplot
        rect = Rectangle((0, 0), 1, 1, transform=axes[idx].transAxes,
                         fill=False, edgecolor=border_color, linewidth=14)
        axes[idx].add_patch(rect)
        rect.set_zorder(10)

        # Class name as bold title (auto-wraps for long names)
        axes[idx].set_title(class_name, fontsize=16, fontweight='bold',
                            color=title_color, pad=15, linespacing=1.3)

    # Main title
    plt.suptitle(title, fontsize=38, fontweight='bold', y=0.96)

    # Adjust layout to prevent overlapping labels and give space for long class names
    plt.subplots_adjust(left=0.15, right=0.85, top=0.90, bottom=0.1,
                        wspace=0.25, hspace=0.25)

    # Save high-resolution gallery
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Success! GradCAM gallery saved: {output_path}")


if __name__ == "__main__":
    make_gradcam_gallery()