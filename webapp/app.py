import streamlit as st
import torch
import timm
from PIL import Image
import os
import random
from torchvision import transforms

# ================= CONFIGURATION =================

# 1. Model Architecture (Must match the one used during training)
MODEL_ARCH = 'swin_base_patch4_window7_224.ms_in22k'

# 2. Path to the trained model weights
MODEL_PATH = 'webapp/model_best.pth.tar'

# 3. Path to the class mapping file
# Expects a text file where each line corresponds to a class name,
# or formatted as "index class_name"
CLASS_MAP_FILE = 'dataset/SO32_class_map.txt' 

# 4. Folder containing test images for demonstration
IMAGE_FOLDER = 'webapp/examples'

# =================================================

st.set_page_config(page_title="Fossil Classification Demo", layout="wide")

def load_class_map(file_path):
    """
    Reads the class_map file and returns a list of class names.
    Supports two common formats:
    1. Plain list: One class name per line (line number = index).
    2. Key-Value pairs: '0 class_name' or 'class_name 0'.
    """
    classes = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Attempt to parse
        is_index_value = False
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            
            parts = line.split(maxsplit=1) 
            
            # If line contains a digit and text, treat as Index-Class pair
            if len(parts) == 2 and parts[0].isdigit():
                classes[int(parts[0])] = parts[1]
                is_index_value = True
            else:
                # Default to line-order index
                classes[idx] = line

        # Convert to list, ensuring continuous indices
        if not classes:
            return []
        
        max_idx = max(classes.keys())
        # Fill missing indices if any
        class_list = [classes.get(i, f"Class_{i}") for i in range(max_idx + 1)]
        return class_list

    except Exception as e:
        st.error(f"Failed to load class map file: {e}")
        return []

@st.cache_resource
def load_model(num_classes):
    """
    Loads the model and caches it to prevent reloading on every interaction.
    """
    try:
        # Create model structure
        model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=num_classes)
        
        # Load weights
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            
            # Handle different checkpoint saving formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dictionary (strict=False to ignore minor mismatches)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model
        else:
            st.error(f"Model weights not found at: {MODEL_PATH}")
            return None
    except Exception as e:
        st.error(f"Failed to build model (Arch: {MODEL_ARCH}): {e}")
        return None

def process_image(image):
    """
    Preprocesses the image for the Swin Transformer (Resize, Tensor, Normalize).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Standard ImageNet normalization mean and std
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Add batch dimension
    return transform(image).unsqueeze(0)

# ================= SIDEBAR =================
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Model Demo", "Project README"])

# Load class names globally
if os.path.exists(CLASS_MAP_FILE):
    CLASS_NAMES = load_class_map(CLASS_MAP_FILE)
    st.sidebar.success(f"Loaded {len(CLASS_NAMES)} classes")
else:
    st.sidebar.error(f"Class map file not found: {CLASS_MAP_FILE}")
    CLASS_NAMES = []

# ================= MAIN PAGE LOGIC =================
if selection == "Project README":
    st.title("📄 Project Introduction")
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("README.md not found.")

elif selection == "Model Demo":
    st.title(f"🦕 Fossil Image Classification")
    st.caption(f"Architecture: `{MODEL_ARCH}`")

    # Initialize Session State
    if 'current_image_path' not in st.session_state:
        st.session_state.current_image_path = None
        st.session_state.prediction = None

    # Get list of images
    image_files = []
    if os.path.exists(IMAGE_FOLDER):
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Select Image")
        if st.button("🎲 Random Selection", use_container_width=True):
            if image_files:
                selected = random.choice(image_files)
                st.session_state.current_image_path = os.path.join(IMAGE_FOLDER, selected)
                # Reset previous prediction
                st.session_state.prediction = None
            else:
                st.warning("Test image folder is empty!")
        
        if st.session_state.current_image_path:
            image = Image.open(st.session_state.current_image_path).convert('RGB')
            st.image(image, caption=os.path.basename(st.session_state.current_image_path), use_container_width=True)

    with col2:
        st.subheader("2. Inference")
        if st.session_state.current_image_path:
            if st.button("🚀 Classify", type="primary", use_container_width=True):
                if not CLASS_NAMES:
                    st.error("Cannot classify: Class map not loaded.")
                else:
                    # Load model (Pass number of classes dynamically)
                    model = load_model(len(CLASS_NAMES))
                    
                    if model:
                        with st.spinner('Running Swin Transformer inference...'):
                            img = Image.open(st.session_state.current_image_path).convert('RGB')
                            input_tensor = process_image(img)
                            
                            # Run inference
                            with torch.no_grad():
                                outputs = model(input_tensor)
                                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                            
                            # Get Top-3 results
                            top_probs, top_ids = torch.topk(probs, 3)
                            
                            st.session_state.prediction = (top_probs, top_ids)

            # Display Results
            if st.session_state.prediction:
                top_probs, top_ids = st.session_state.prediction
                
                # Best Prediction
                best_idx = top_ids[0].item()
                best_conf = top_probs[0].item()
                best_name = CLASS_NAMES[best_idx] if best_idx < len(CLASS_NAMES) else f"Unknown ID {best_idx}"
                
                st.success(f"### Result: {best_name}")
                st.metric("Confidence", f"{best_conf*100:.2f}%")
                
                # Detailed Probabilities (Top 3)
                st.write("---")
                st.write("**Top 3 Predictions:**")
                for i in range(3):
                    idx = top_ids[i].item()
                    conf = top_probs[i].item()
                    name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"ID {idx}"
                    st.progress(conf, text=f"{name} ({conf*100:.1f}%)")

        else:
            st.info("Please select an image on the left first.")