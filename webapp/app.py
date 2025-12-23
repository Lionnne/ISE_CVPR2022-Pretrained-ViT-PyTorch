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
    Replaces relative image paths in BOTH Markdown and HTML tags 
    with absolute GitHub Raw URLs.
    """
    base_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"
    
    # 1. Fix standard Markdown images: ![alt](path)
    def replacer_md(match):
        alt_text = match.group(1)
        path = match.group(2)
        if path.startswith("http"): return match.group(0)
        return f"![{alt_text}]({base_url}{path})"
    
    content = re.sub(r'!\[(.*?)\]\((.*?)\)', replacer_md, content)

    # 2. Fix HTML images: src="path"
    # This matches src="path" or src='path'
    def replacer_html(match):
        quote = match.group(1) # Capture the opening quote (" or ')
        path = match.group(2)  # Capture the path inside
        if path.startswith("http"): return match.group(0)
        # Reconstruct the string with the new full URL
        return f'src={quote}{base_url}{path}{quote}'

    content = re.sub(r'src=(["\'])(.*?)\1', replacer_html, content)

    return content
    
# ================= CONFIGURATION =================

# 1. Model Architecture
MODEL_ARCH = 'swin_base_patch4_window7_224.ms_in22k'

# 2. Google Drive File ID
GDRIVE_FILE_ID = '1mKtniwNyszW1ynar3dQbumRE4OPUs5VS' 

# 3. Local paths
MODEL_PATH = 'webapp/model_best.pth.tar'
CLASS_MAP_FILE = 'dataset/SO32_class_map.txt' 
IMAGE_FOLDER = 'webapp/examples'

# ================= DEVICE CONFIGURATION =================
# Automatically detect if a GPU is available
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = torch.device('cpu')
    DEVICE_NAME = "CPU"

# =================================================

st.set_page_config(page_title="Fossil Classification Demo", layout="wide")

def download_model_if_needed():
    """Checks if the model exists locally. If not, downloads it from Google Drive."""
    if not os.path.exists(MODEL_PATH):
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
    """Reads the class_map file and returns a list of class names."""
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
                classes[idx] = line

        if not classes: return []
        
        max_idx = max(classes.keys())
        return [classes.get(i, f"Class_{i}") for i in range(max_idx + 1)]

    except Exception as e:
        st.error(f"Failed to load class map file: {e}")
        return []

@st.cache_resource
def load_model(num_classes):
    """Loads the model architecture and weights."""
    if not download_model_if_needed():
        return None

    try:
        model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=num_classes)
        
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
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
    """Preprocesses the image: Resize -> Tensor -> Normalize."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# ================= SIDEBAR & SETUP =================
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Model Demo", "Project README"])

status_icon = "🚀" if "cuda" in str(DEVICE) else "💻"
st.sidebar.markdown(f"**Hardware:** {DEVICE_NAME} {status_icon}")

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
            fixed_content = fix_readme_images(content)
            st.markdown(fixed_content, unsafe_allow_html=True)
    else:
        st.warning("README.md not found in the current directory.")

elif selection == "Model Demo":
    st.title(f"Microfossil Classification")
    st.caption(f"Model Architecture: `{MODEL_ARCH}` | Running on: `{DEVICE_NAME}`")

    # Initialize Session State
    if 'selected_mode' not in st.session_state:
        st.session_state.selected_mode = 'random' # 'random' or 'upload'
    if 'current_image_path' not in st.session_state:
        st.session_state.current_image_path = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None

    # Load images list
    image_files = []
    if os.path.exists(IMAGE_FOLDER):
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # Layout: 2 Columns
    col1, col2 = st.columns([1, 1])

    # --- Column 1: Image Selection (Modified to support Tabs) ---
    with col1:
        st.subheader("1. Select Image")
        
        # tabs
        tab1, tab2 = st.tabs(["🎲 Random Sample", "📤 Upload Image"])
        
        # === TAB 1: Random Selection ===
        with tab1:
            if st.button("Pick Random Image", use_container_width=True):
                if image_files:
                    selected = random.choice(image_files)
                    st.session_state.current_image_path = os.path.join(IMAGE_FOLDER, selected)
                    st.session_state.selected_mode = 'random'
                    st.session_state.prediction = None
                else:
                    st.warning(f"No images found in '{IMAGE_FOLDER}'")

            if st.session_state.selected_mode == 'random' and st.session_state.current_image_path:
                st.info(f"Selected: {os.path.basename(st.session_state.current_image_path)}")

        # === TAB 2: Upload Image ===
        with tab2:
            uploaded_file = st.file_uploader("Choose a file...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                st.session_state.uploaded_file = uploaded_file
                st.session_state.selected_mode = 'upload'
                # st.session_state.prediction = None 

        # --- Display the image based on mode ---
        display_img = None
        try:
            if st.session_state.selected_mode == 'random' and st.session_state.current_image_path:
                display_img = Image.open(st.session_state.current_image_path).convert('RGB')
                st.image(display_img, caption="Random Sample", use_container_width=True)
            
            elif st.session_state.selected_mode == 'upload' and st.session_state.uploaded_file:
                display_img = Image.open(st.session_state.uploaded_file).convert('RGB')
                st.image(display_img, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")


    # --- Column 2: Inference (Updated to handle both sources) ---
    with col2:
        st.subheader("2. Inference")
        
        # Check if we have a valid image to classify
        has_valid_image = (st.session_state.selected_mode == 'random' and st.session_state.current_image_path) or \
                          (st.session_state.selected_mode == 'upload' and st.session_state.uploaded_file)

        if has_valid_image:
            if st.button("🚀 Classify", type="primary", use_container_width=True):
                if not CLASS_NAMES:
                    st.error("Cannot classify: Class map not loaded.")
                else:
                    # Load model
                    model = load_model(len(CLASS_NAMES))
                    
                    if model:
                        with st.spinner(f'Analyzing image on {DEVICE_NAME}...'):
                            try:
                                # Determine which image to open
                                if st.session_state.selected_mode == 'random':
                                    img = Image.open(st.session_state.current_image_path).convert('RGB')
                                else:
                                    img = Image.open(st.session_state.uploaded_file).convert('RGB')

                                input_tensor = process_image(img)
                                
                                # Move input data to device
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
                    
                    st.progress(conf, text=f"#{i+1} {name} ({conf*100:.2f}%)")

        else:
            st.info("Please select or upload an image on the left first.")
