import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# Add model directory to path so we can import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
import config

# --- Page Configuration ---
st.set_page_config(
    page_title="Deepfake Detector | Facial Authenticity Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Modern UI Styling (CSS) ---
st.markdown("""
    <style>
    /* Main Layout */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Style native Streamlit containers as cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1e2130 !important;
        border: 1px solid #3d4156 !important;
        border-radius: 15px !important;
        padding: 20px !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3) !important;
    }
    
    /* Result Styling */
    .result-header {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .real-text { color: #2ecc71; }
    .fake-text { color: #e74c3c; }
    
    .confidence-label {
        font-size: 1.2rem;
        color: #a3a8b4;
        margin-bottom: 20px;
    }
    
    /* Footer Styling */
    .footer {
        text-align: center;
        padding: 40px 20px;
        margin-top: 50px;
        color: #6c757d;
        border-top: 1px solid #3d4156;
    }
    .footer a {
        color: #a3a8b4;
        text-decoration: none;
        margin: 0 15px;
        transition: 0.3s;
        font-size: 1.1rem;
    }
    .footer a:hover {
        color: #2ecc71;
    }
    .footer .disclaimer {
        font-size: 0.85rem;
        font-style: italic;
        margin-top: 15px;
    }
    .footer .copyright {
        margin-top: 10px;
        font-size: 0.9rem;
    }
    
    /* Hide Sidebars and Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    
    /* Buttons */
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #27ae60;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Core Model Logic ---
@st.cache_resource
def load_model():
    model_path = config.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        return None

def predict(model, image):
    img = image.resize(config.IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]
    
    if prediction < 0.5:
        label, conf, css_class = "REAL", (1 - prediction) * 100, "real-text"
    else:
        label, conf, css_class = "FAKE", prediction * 100, "fake-text"
        
    return label, conf, css_class

# --- Main Application UI ---
model = load_model()

# Centered layout using columns
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    # Header
    st.markdown("""
        <div style='text-align: center; margin-bottom: 40px;'>
            <h1 style='font-size: 3rem; margin-bottom: 0;'>🛡️ Deepfake Detector</h1>
            <p style='color: #a3a8b4; font-size: 1.2rem;'>AI-powered facial authenticity analysis</p>
        </div>
    """, unsafe_allow_html=True)

    # Input Section Card
    with st.container(border=True):
        st.subheader("📸 Biometric Capture")
        st.info("Upload a face image to check if it is real or AI-generated.")
        
        option = st.segmented_control("Input Source", ["📤 Upload Image", "📷 Live Webcam"], default="📤 Upload Image")
        
        input_image = None
        if option == "📤 Upload Image":
            uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                input_image = Image.open(uploaded_file).convert('RGB')
        else:
            camera_photo = st.camera_input("")
            if camera_photo:
                input_image = Image.open(camera_photo).convert('RGB')

    # Preview & Analysis Section
    if input_image:
        st.write("") # Spacer
        with st.container(border=True):
            st.subheader(" Analysis Preview")
            st.image(input_image, use_container_width=True, caption="Biometric Sample")
            
            if model is not None:
                if st.button("Analyze Authenticity"):
                    with st.spinner("Analyzing neural patterns..."):
                        label, confidence, css_class = predict(model, input_image)
                        
                        # Result Display
                        icon = "✅" if label == "REAL" else "❌"
                        st.markdown(f"""
                            <div style='margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 12px;'>
                                <div class='result-header {css_class}'>{icon} {label}</div>
                                <div class='confidence-label'>Confidence Score: {confidence:.2f}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(float(confidence) / 100)
                        
                        if label == "FAKE":
                            st.error("Adversarial artifacts detected. This image likely originated from an AI generator.")
                        else:
                            st.success("No synthetic patterns detected. This image appears to be an authentic photograph.")
            else:
                st.warning("Analysis engine unavailable. Please ensure the model is trained.")

# --- Professional Footer ---
st.markdown(f"""
    <div class='footer'>
        <div style='margin-bottom: 20px;'>
            <h3 style='color: #ffffff; margin-bottom: 5px;'>Deepfake Detector</h3>
            <p>AI-powered facial authenticity analysis</p>
        </div>
        <div style='margin-bottom: 20px;'>
            <a href='mailto:banikesha03@gmail.com' target='_blank'>📧 Gmail</a>
            <a href='https://www.linkedin.com/in/eshanibanik/' target='_blank'>🔗 LinkedIn</a>
        </div>
        <div class='disclaimer'>
            For educational/demo purposes only. AI detection is diagnostic, not definitive.
        </div>
        <div class='copyright'>
            © 2026 Eshani Banik | Built with TensorFlow & EfficientNetB0
        </div>
    </div>
""", unsafe_allow_html=True)
