"""
Streamlit Web Application for Traffic Sign Recognition
Upload an image and get real-time predictions
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# Class names dictionary
CLASS_NAMES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 32: 'End speed + passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}

@st.cache_resource
def load_model(model_path):
    """Load trained model (cached for performance)"""
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Resize to 32x32
    img_resized = cv2.resize(img_array, (32, 32))
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_image(model, image):
    """Make prediction on image"""
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Predict
    predictions = model.predict(processed_img, verbose=0)
    
    # Get top 5 predictions
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_probs = predictions[0][top_5_indices]
    
    return top_5_indices, top_5_probs

def main():
    # Title and description
    st.title("🚦 Traffic Sign Recognition System")
    st.markdown("""
    Upload an image of a traffic sign and the AI model will classify it!
    This model is trained on the **GTSRB dataset** with **43 different traffic sign classes**.
    """)
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("""
    **Model Details:**
    - Architecture: Deep CNN
    - Dataset: GTSRB (German Traffic Sign Recognition Benchmark)
    - Classes: 43 different traffic signs
    - Input Size: 32x32 pixels
    
    **Features:**
    - Real-time prediction
    - Top-5 confidence scores
    - Support for multiple image formats
    """)
    
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose Model:",
        ["deep_cnn_best.h5", "simple_cnn_best.h5", "mobilenet_transfer_best.h5"]
    )
    
    # Load model
    model_path = f"models/{model_choice}"
    model = load_model(model_path)
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first.")
        st.stop()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a traffic sign image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a traffic sign"
        )
        
        # Sample images option
        use_sample = st.checkbox("Use sample images")
        
        if use_sample:
            sample_images = {
                "Stop Sign": "sample_images/stop.png",
                "Speed Limit 50": "sample_images/speed_50.png",
                "Yield": "sample_images/yield.png"
            }
            selected_sample = st.selectbox("Select sample:", list(sample_images.keys()))
            
            if st.button("Load Sample"):
                uploaded_file = sample_images[selected_sample]
        
        if uploaded_file is not None:
            # Load image
            if isinstance(uploaded_file, str):
                image = Image.open(uploaded_file)
            else:
                image = Image.open(uploaded_file)
            
            # Display uploaded image
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Predict button
            if st.button("🔍 Classify Sign", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    top_5_indices, top_5_probs = predict_image(model, image)
                    
                    # Store results in session state
                    st.session_state['predictions'] = (top_5_indices, top_5_probs)
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        if 'predictions' in st.session_state:
            top_5_indices, top_5_probs = st.session_state['predictions']
            
            # Display top prediction with large text
            top_class = CLASS_NAMES[top_5_indices[0]]
            top_confidence = top_5_probs[0] * 100
            
            st.markdown(f"""
            <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; text-align: center;'>
                <h2 style='color: #1f77b4; margin: 0;'>Predicted Sign:</h2>
                <h1 style='color: #2ca02c; margin: 10px 0;'>{top_class}</h1>
                <h3 style='color: #666; margin: 0;'>Confidence: {top_confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display top 5 predictions
            st.subheader("Top 5 Predictions:")
            
            for i, (idx, prob) in enumerate(zip(top_5_indices, top_5_probs), 1):
                class_name = CLASS_NAMES[idx]
                confidence = prob * 100
                
                # Progress bar for confidence
                st.markdown(f"**{i}. {class_name}**")
                st.progress(float(prob))
                st.caption(f"Confidence: {confidence:.2f}%")
                st.markdown("")
            
            # Confidence visualization
            st.subheader("Confidence Distribution:")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = [CLASS_NAMES[idx][:20] for idx in top_5_indices]
            ax.barh(labels, top_5_probs * 100, color='steelblue')
            ax.set_xlabel('Confidence (%)')
            ax.set_title('Top 5 Predictions')
            ax.invert_yaxis()
            
            for i, v in enumerate(top_5_probs * 100):
                ax.text(v + 1, i, f'{v:.1f}%', va='center')
            
            st.pyplot(fig)
            
        else:
            st.info("👆 Upload an image and click 'Classify Sign' to see predictions")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
        <p>Model trained on GTSRB dataset | 43 Traffic Sign Classes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()