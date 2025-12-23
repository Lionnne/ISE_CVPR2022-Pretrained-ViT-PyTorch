import streamlit as st
import torch
import timm
from PIL import Image
import os
import random
from torchvision import transforms
import gdown  # pip install gdown
import re

GITHUB_USER = "Lionnne"
GITHUB_REPO = "ISE_CVPR2022-Pretrained-ViT-PyTorch"
GITHUB_BRANCH = "main"

def fix_readme_images(content):
    """
    Replaces relative image paths (e.g., 'figs/plot.png') 
    with absolute GitHub Raw URLs so they display correctly in Streamlit.
    """
    base_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"
    
    # Regex to find markdown images: ![alt](path)
    def replacer(match):
        alt_text = match.group(1)
        path = match.group(2)
        # If it's already a web link (starts with http), leave it alone
        if path.startswith("http"):
            return match.group(0)
        # Otherwise, prepend the GitHub Raw URL
        return f"![{alt_text}]({base_url}{path})"

    return re.sub(r'!\[(.*?)\]\((.*?)\)', replacer, content)
    
# ================= CONFIGURATION =================

# 1. Model Architecture
# Must match the architecture used during training
MODEL_ARCH = 'swin_base_patch4_window7_224.ms_in22k'

# 2. Google Drive File ID (IMPORTANT: Replace with your actual File ID)
# Link format: https://drive.google.com/file/d/YOUR_ID_HERE/view
GDRIVE_FILE_ID = '1mKtniwNyszW1ynar3dQbumRE4OPUs5VS' 

# 3. Local paths
MODEL_PATH = 'webapp/model_best.pth.tar'
CLASS_MAP_FILE = 'dataset/SO32_class_map.txt' 
IMAGE_FOLDER = 'webapp/examples'

# ================= DEVICE CONFIGURATION =================
# Automatically detect if a GPU is available
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    # Get the specific name of the GPU (e.g., "NVIDIA GeForce RTX 4090")
    DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = torch.device('cpu')
    DEVICE_NAME = "CPU"

# =================================================

st.set_page_config(page_title="Fossil Classification Demo", layout="wide")

def download_model_if_needed():
    """
    Checks if the model exists locally. If not, downloads it from Google Drive.
    """
    if not os.path.exists(MODEL_PATH):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        st.info(f"Model weights not found locally. Downloading from Google Drive...")
        try:
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
            st.success("Download complete!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.warning("Please check if the Google Drive File ID is correct and has 'Anyone with the link' permission.")
            return False
    return True

def load_class_map(file_path):
    """
    Reads the class_map file and returns a list of class names.
    Supports 'Index Name' format or simple line-by-line list.
    """
    classes = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            
            parts = line.split(maxsplit=1) 
            # Check for "0 ClassName" format
            if len(parts) == 2 and parts[0].isdigit():
                classes[int(parts[0])] = parts[1]
            else:
                # Default to line index
                classes[idx] = line

        if not classes: return []
        
        max_idx = max(classes.keys())
        # Reconstruct list ensuring correct order
        return [classes.get(i, f"Class_{i}") for i in range(max_idx + 1)]

    except Exception as e:
        st.error(f"Failed to load class map file: {e}")
        return []

@st.cache_resource
def load_model(num_classes):
    """
    Loads the model architecture and weights.
    Handles  transfer (CPU <-> GPU).
    """
    # 1. Download if missing
    if not download_model_if_needed():
        return None

    try:
        # 2. Create Model Architecture
        model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=num_classes)
        
        # 3. Load Weights
        if os.path.exists(MODEL_PATH):
            # Load to CPU first to avoid CUDA OOM or device mismatch during load
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            
            # Handle different saving formats (dict vs model object)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            
            # 4. Move to the detected device (GPU or CPU)
            model.to(DEVICE)
            model.eval()
            return model
        else:
            st.error(f"Model file missing at {MODEL_PATH}")
            return None
    except Exception as e:
        st.error(f"Failed to build model: {e}")
        return None

def process_image(image):
    """
    Preprocesses the image: Resize -> Tensor -> Normalize.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Standard ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0) # Add batch dimension (1, C, H, W)

# ================= SIDEBAR & SETUP =================
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Model Demo", "Project README"])

# Display Device Status
status_icon = "🚀" if "cuda" in str(DEVICE) else "💻"
st.sidebar.markdown(f"**Hardware:** {DEVICE_NAME} {status_icon}")

# Load Class Names
if os.path.exists(CLASS_MAP_FILE):
    CLASS_NAMES = load_class_map(CLASS_MAP_FILE)
    st.sidebar.success(f"Loaded {len(CLASS_NAMES)} classes")
else:
    st.sidebar.error(f"Missing file: {CLASS_MAP_FILE}")
    CLASS_NAMES = []

# ================= MAIN LOGIC =================

if selection == "Project README":
    st.title("📄 Project Introduction")
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
            # Apply the fix function before displaying
            fixed_content = fix_readme_images(content)
            st.markdown(fixed_content)
    else:
        st.warning("README.md not found in the current directory.")

elif selection == "Model Demo":
    st.title(f"Microfossil Classification")
    st.caption(f"Model Architecture: `{MODEL_ARCH}` | Running on: `{device_name}`")

    # Initialize Session State
    if 'current_image_path' not in st.session_state:
        st.session_state.current_image_path = None
        st.session_state.prediction = None

    # Load images
    image_files = []
    if os.path.exists(IMAGE_FOLDER):
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # Layout: 2 Columns
    col1, col2 = st.columns([1, 1])

    # --- Column 1: Image Selection ---
    with col1:
        st.subheader("1. Select Image")
        if st.button("🎲 Random Selection", use_container_width=True):
            if image_files:
                selected = random.choice(image_files)
                st.session_state.current_image_path = os.path.join(IMAGE_FOLDER, selected)
                st.session_state.prediction = None # Reset result
            else:
                st.warning(f"No images found in '{IMAGE_FOLDER}'")
        
        if st.session_state.current_image_path:
            image = Image.open(st.session_state.current_image_path).convert('RGB')
            st.image(image, caption=os.path.basename(st.session_state.current_image_path), use_container_width=True)

    # --- Column 2: Inference ---
    with col2:
        st.subheader("2. Inference")
        if st.session_state.current_image_path:
            if st.button("🚀 Classify", type="primary", use_container_width=True):
                if not CLASS_NAMES:
                    st.error("Cannot classify: Class map not loaded.")
                else:
                    # Load model
                    model = load_model(len(CLASS_NAMES))
                    
                    if model:
                        with st.spinner(f'Analyzing image on {device_name}...'):
                            try:
                                img = Image.open(st.session_state.current_image_path).convert('RGB')
                                input_tensor = process_image(img)
                                
                                # Move input data to the same device as the model (GPU or CPU)
                                input_tensor = input_tensor.to(DEVICE)
                                
                                with torch.no_grad():
                                    outputs = model(input_tensor)
                                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                                
                                # Get Top-5 Results
                                top_k = 5
                                actual_k = min(top_k, len(CLASS_NAMES))
                                top_probs, top_ids = torch.topk(probs, actual_k)
                                
                                st.session_state.prediction = (top_probs, top_ids, actual_k)
                            except Exception as e:
                                st.error(f"Inference error: {e}")

            # Display Results
            if st.session_state.prediction:
                top_probs, top_ids, k = st.session_state.prediction
                
                # Best Result
                best_idx = top_ids[0].item()
                best_conf = top_probs[0].item()
                best_name = CLASS_NAMES[best_idx] if best_idx < len(CLASS_NAMES) else f"Unknown ID {best_idx}"
                
                st.success(f"### Top 1: {best_name}")
                st.metric("Confidence", f"{best_conf*100:.2f}%")
                
                st.write("---")
                st.write(f"**Top {k} Probabilities:**")
                
                # Loop to display progress bars
                for i in range(k):
                    idx = top_ids[i].item()
                    conf = top_probs[i].item()
                    name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
                    
                    # Display progress bar with label
                    st.progress(conf, text=f"#{i+1} {name} ({conf*100:.2f}%)")

        else:
            st.info("Please select an image on the left first.")
