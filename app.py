import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import os
import sys

# Add src to path
sys.path.append('src')
from data_loader import GTSRBDataLoader
from preprocessor import ImagePreprocessor

# Page configuration
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    model_dir = 'models'
    
    if os.path.exists(f'{model_dir}/custom_cnn_best.h5'):
        models['Custom CNN'] = load_model(f'{model_dir}/custom_cnn_best.h5')
    
    if os.path.exists(f'{model_dir}/mobilenet_best.h5'):
        models['MobileNet V2'] = load_model(f'{model_dir}/mobilenet_best.h5')
    
    return models

@st.cache_resource
def load_class_info():
    """Load class names and information"""
    loader = GTSRBDataLoader('data')
    return loader.class_names

def preprocess_image(image, target_size=(32, 32)):
    """Preprocess uploaded image"""
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize
    img_resized = cv2.resize(img_array, target_size)
    
    # Normalize
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_resized

def predict_sign(model, image):
    """Make prediction on image"""
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Get top 5 predictions
    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    top_5_probs = predictions[0][top_5_idx]
    
    return predicted_class, confidence, top_5_idx, top_5_probs

# Main app
def main():
    st.markdown('<p class="main-header">🚦 Traffic Sign Recognition System</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning Based Traffic Sign Classifier</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("❌ No trained models found! Please train models first using train.py")
        st.info("Run: `python src/train.py`")
        return
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        list(models.keys())
    )
    
    selected_model = models[model_name]
    
    # Load class names
    class_names = load_class_info()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.info(f"""
    **Selected Model:** {model_name}
    
    **Number of Classes:** 43
    
    **Input Size:** 32x32x3
    """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📈 Performance", "ℹ️ About"])
    
    with tab1:
        st.header("Upload and Classify Traffic Sign")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg', 'ppm'],
                help="Upload a traffic sign image"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_container_width=True)
                
                # Predict button
                if st.button("🚀 Classify", type="primary"):
                    with st.spinner('Analyzing...'):
                        # Preprocess
                        processed_img, resized_img = preprocess_image(image)
                        
                        # Predict
                        pred_class, confidence, top_5_idx, top_5_probs = predict_sign(
                            selected_model, processed_img
                        )
                        
                        # Store in session state
                        st.session_state['prediction'] = {
                            'class': pred_class,
                            'confidence': confidence,
                            'top_5_idx': top_5_idx,
                            'top_5_probs': top_5_probs,
                            'resized_img': resized_img
                        }
        
        with col2:
            if 'prediction' in st.session_state:
                pred = st.session_state['prediction']
                
                # Display result
                st.markdown("### 🎯 Prediction Result")
                
                # Predicted class with proper name
                sign_name = class_names.get(pred['class'], f'Unknown Sign (Class {pred["class"]})')
                st.success(f"**Predicted Sign:** {sign_name}")
                
                # Confidence
                st.metric(
                    label="Confidence",
                    value=f"{pred['confidence']*100:.2f}%"
                )
                
                # Confidence bar
                conf_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred['confidence']*100,
                    title={'text': "Confidence Level"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "lightblue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                conf_fig.update_layout(height=250)
                st.plotly_chart(conf_fig, use_container_width=True)
                
                # Top 5 predictions
                st.markdown("### 📊 Top 5 Predictions")
                
                top_5_df = pd.DataFrame({
                    'Class': [class_names.get(idx, f'Unknown Sign (Class {idx})') 
                             for idx in pred['top_5_idx']],
                    'Probability': pred['top_5_probs']
                })
                
                fig = px.bar(
                    top_5_df,
                    x='Probability',
                    y='Class',
                    orientation='h',
                    color='Probability',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    showlegend=False,
                    height=300,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show preprocessed image
                st.markdown("### 🖼️ Preprocessed Image")
                st.image(pred['resized_img'], caption='32x32 Resized', width=200)
    
    with tab2:
        st.header("Model Performance Metrics")
        
        # Model selection for performance view
        perf_model_name = model_name.lower().replace(' ', '_')
        
        # Check if performance metrics exist
        metrics_file = f"models/{perf_model_name}_metrics.csv"
        
        if os.path.exists(metrics_file):
            # Load and display metrics
            metrics_df = pd.read_csv(metrics_file)
            
            st.success("✅ Evaluation metrics loaded successfully!")
            st.markdown("---")
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                acc_val = metrics_df['accuracy'].values[0]
                st.metric("🎯 Accuracy", f"{acc_val:.2%}", 
                         delta=f"{(acc_val - 0.5) * 100:.1f}%")
            with col2:
                prec_val = metrics_df['precision'].values[0]
                st.metric("🎪 Precision", f"{prec_val:.2%}",
                         delta=f"{(prec_val - 0.5) * 100:.1f}%")
            with col3:
                rec_val = metrics_df['recall'].values[0]
                st.metric("🔄 Recall", f"{rec_val:.2%}",
                         delta=f"{(rec_val - 0.5) * 100:.1f}%")
            with col4:
                f1_val = metrics_df['f1_score'].values[0]
                st.metric("⚖️ F1-Score", f"{f1_val:.2%}",
                         delta=f"{(f1_val - 0.5) * 100:.1f}%")
            
            st.markdown("---")
            
            # Metrics comparison chart
            st.markdown("### 📊 Metrics Overview")
            metrics_chart_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Score': [acc_val, prec_val, rec_val, f1_val]
            })
            
            fig_metrics = px.bar(
                metrics_chart_df,
                x='Metric',
                y='Score',
                color='Score',
                color_continuous_scale='Viridis',
                text='Score'
            )
            fig_metrics.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig_metrics.update_layout(
                yaxis_range=[0, 1.1],
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
            
        else:
            st.warning("⚠️ Performance metrics not available for this model.")
            st.info("📊 Run the evaluation script to generate metrics.")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.code("python src/evaluate.py", language="bash")
            with col2:
                if st.button("🚀 Run Evaluation Now", type="primary"):
                    with st.spinner("Running evaluation... This may take a few minutes."):
                        try:
                            import subprocess
                            result = subprocess.run(
                                ['python', 'src/evaluate.py'],
                                capture_output=True,
                                text=True,
                                timeout=600
                            )
                            if result.returncode == 0:
                                st.success("✅ Evaluation completed! Refresh the page to see results.")
                                st.balloons()
                            else:
                                st.error(f"❌ Evaluation failed: {result.stderr}")
                        except subprocess.TimeoutExpired:
                            st.error("⏱️ Evaluation timed out. Please run manually.")
                        except Exception as e:
                            st.error(f"❌ Error running evaluation: {str(e)}")
        
        st.markdown("---")
        
        # Show training history if available
        history_img = f"models/{perf_model_name}_history.png"
        if os.path.exists(history_img):
            st.markdown("### 📈 Training History")
            st.image(history_img, use_container_width=True)
        else:
            st.info("📈 Training history plot not available.")
        
        st.markdown("---")
        
        # Show confusion matrix if available
        cm_img = f"models/{perf_model_name}_confusion_matrix.png"
        if os.path.exists(cm_img):
            st.markdown("### 🎯 Confusion Matrix")
            st.image(cm_img, use_container_width=True)
        else:
            st.info("🎯 Confusion matrix not available.")
        
        st.markdown("---")
        
        # Show per-class accuracy if available
        per_class_img = f"models/{perf_model_name}_per_class.png"
        if os.path.exists(per_class_img):
            st.markdown("### 📊 Per-Class Accuracy")
            st.image(per_class_img, use_container_width=True)
        else:
            st.info("📊 Per-class accuracy plot not available.")
        
        # Model comparison if both models exist
        if os.path.exists('models/model_comparison.png'):
            st.markdown("---")
            st.markdown("### 🔬 Model Comparison")
            st.image('models/model_comparison.png', use_container_width=True)
    
    with tab3:
        st.header("About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This is a **Traffic Sign Recognition System** built using Deep Learning techniques.
        The system can classify 43 different types of traffic signs from the German 
        Traffic Sign Recognition Benchmark (GTSRB) dataset.
        
        ### 🔧 Technical Stack
        
        - **Deep Learning Framework:** TensorFlow/Keras
        - **Computer Vision:** OpenCV
        - **Web Framework:** Streamlit
        - **Data Processing:** NumPy, Pandas
        - **Visualization:** Matplotlib, Seaborn, Plotly
        
        ### 🏗️ Architecture
        
        #### Custom CNN
        - 3 Convolutional blocks with BatchNormalization
        - Max Pooling and Dropout for regularization
        - Fully connected layers with 512 and 256 neurons
        
        #### MobileNetV2 (Transfer Learning)
        - Pre-trained on ImageNet
        - Custom classification head
        - Frozen base model for efficiency
        
        ### ✨ Features
        
        1. **Image Preprocessing**
           - Resizing to 32x32
           - Normalization
           - Optional CLAHE enhancement
        
        2. **Data Augmentation** (Bonus)
           - Rotation, shifting, zooming
           - Brightness adjustment
           - Improved model generalization
        
        3. **Model Comparison** (Bonus)
           - Custom CNN vs. MobileNetV2
           - Performance metrics comparison
        
        4. **Interactive Web Interface**
           - Real-time prediction
           - Confidence visualization
           - Top-5 predictions
        
        ### 📊 Dataset
        
        **GTSRB (German Traffic Sign Recognition Benchmark)**
        - 43 classes of traffic signs
        - ~50,000 images total
        - Real-world driving conditions
        
        ### 👨‍💻 Developer
        
        ML Internship Task 8 - Traffic Sign Recognition
        """)
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Start")
        
        st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models
python src/train.py

# 3. Evaluate models
python src/evaluate.py

# 4. Run Streamlit app
streamlit run app.py
        """, language="bash")

if __name__ == "__main__":
    main()