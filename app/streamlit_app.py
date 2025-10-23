"""
Streamlit Web Application for Traffic Sign Recognition
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ===== FIX IMPORT PATH (Crucial for running Streamlit from app/ directory) =====
# Get the project root directory (e.g., D:\ML\Traffic-Sign-Recognition-clean)
# This assumes the script is inside app/ or similar.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add to Python path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import project modules
import_error = None
# --- Assume the following files exist under the project root: ---
# D:\ML\Traffic-Sign-Recognition-clean\src\utils.py
# D:\ML\Traffic-Sign-Recognition-clean\src\config.py
import_error = None
try:
    from src.utils import load_model, predict_sign, get_class_name
    # Keep config, as it is needed for config.MODEL_DIR later
    from src import config 
except ImportError as e:
    # If it's a 'config' error, try loading paths manually to proceed
    if 'config' in str(e):
         # Hardcode the essential paths needed by the Streamlit app
         class ConfigMock:
             MODEL_DIR = os.path.join(project_root, 'models', 'saved_models')
             CLASS_NAMES = {} # Only used in Tab 2/4 which can be handled by utils
         config = ConfigMock()
         import_error = str(e) + "\n(Using hardcoded paths to bypass module load failure)"
    else:
        import_error = str(e)
# Page configuration
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show import error if any (This block is kept to inform the user)
if import_error:
    st.error("❌ **Cannot import project modules**")
    st.code(import_error)
    
    with st.expander("🔍 Debug Information"):
        st.write("**Current Working Directory:**")
        st.code(os.getcwd())
        
        st.write("**Project Root:**")
        st.code(project_root)
        
        st.write("**Python Path (first 5):**")
        for i, path in enumerate(sys.path[:5], 1):
            st.code(f"{i}. {path}")
        
        st.write("**Expected src/ location:**")
        src_path = os.path.join(project_root, 'src')
        st.code(src_path)
        st.write(f"Exists: {os.path.exists(src_path)}")
        
        if os.path.exists(src_path):
            st.write("**Files in src/:**")
            files = os.listdir(src_path)
            st.write(files)
    
    st.info("""
    **Solutions (External Fixes):**
    
    1. **Verify Files:** Ensure the files `src/config.py` and `src/utils.py` exist.
    2. **Run Correctly:** Execute Streamlit from the project root:
        ```bash
        cd D:\ML\Traffic-Sign-Recognition-clean
        streamlit run app/streamlit_app.py
        ```
    3. **Set PYTHONPATH:** (If necessary)
        ```bash
        $env:PYTHONPATH = "D:\ML\Traffic-Sign-Recognition-clean"
        streamlit run app/streamlit_app.py
        ```
    """)
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #4ECDC4;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚦 Traffic Sign Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Classify traffic signs using Deep Learning (GTSRB Dataset)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["Custom CNN", "MobileNet (Transfer Learning)"],
        help="Choose which model to use for prediction"
    )
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for a valid prediction"
    )
    
    st.markdown("---")
    
    # Information
    st.header("ℹ️ About")
    st.info("""
    **GTSRB Dataset**
    - 43 traffic sign classes
    - German Traffic Sign Recognition Benchmark
    - Trained with data augmentation
    
    **Models Available:**
    - Custom CNN (~500K params)
    - MobileNet (~2M params, transfer learning)
    """)
    
    st.markdown("---")
    
    # Instructions
    st.header("📋 Instructions")
    st.markdown("""
    1. Select a model
    2. Upload a traffic sign image
    3. View prediction results
    4. Explore model performance metrics
    """)

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Prediction",
    "📊 Model Performance", 
    "🎨 Data Augmentation",
    "⚖️ Model Comparison"
])

# TAB 1: PREDICTION
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a traffic sign image",
            type=['jpg', 'jpeg', 'png', 'ppm'],
            help="Upload a clear image of a traffic sign"
        )
        
        image_np = None
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Convert to RGB if needed (handles grayscale and RGBA)
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if uploaded_file is not None:
            try:
                # Determine model path
                if model_type == "Custom CNN":
                    model_path = os.path.join(config.MODEL_DIR, "traffic_sign_cnn.h5")
                else:
                    model_path = os.path.join(config.MODEL_DIR, "traffic_sign_mobilenet.h5")
                
                # Check if model exists
                if not os.path.exists(model_path):
                    st.error(f"❌ Model not found: `{os.path.basename(model_path)}`")
                    st.info(f"""
                    **Train the model first:**
                    ```bash
                    python src/train.py --model {'cnn' if model_type == 'Custom CNN' else 'mobilenet'}
                    ```
                    Or train both models:
                    ```bash
                    python src/train.py --model both
                    ```
                    """)
                else:
                    # Load model
                    with st.spinner("Loading model..."):
                        # Use Streamlit's cache for efficient model loading
                        @st.cache_resource
                        def get_model(path):
                            return load_model(path)
                        
                        model = get_model(model_path)
                    
                    # Make prediction
                    with st.spinner("Analyzing image..."):
                        # Ensure image_np is RGB for prediction (already done in upload block)
                        class_id, confidence, all_predictions = predict_sign(model, image_np)
                        class_name = get_class_name(class_id)
                    
                    # Display results
                    if confidence >= confidence_threshold:
                        st.success(f"✅ **Prediction:** {class_name}")
                        
                        # Metrics
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Class ID", f"{class_id}")
                        with col_b:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        # Top 5 predictions
                        st.markdown("#### 📊 Top 5 Predictions")
                        top_5_indices = np.argsort(all_predictions)[-5:][::-1]
                        
                        for idx in top_5_indices:
                            prob = all_predictions[idx]
                            name = get_class_name(idx)
                            st.progress(float(prob))
                            st.caption(f"{name} - {prob*100:.2f}%")
                    
                    else:
                        st.warning(f"⚠️ Low confidence prediction")
                        st.write(f"**Predicted:** {class_name}")
                        st.write(f"**Confidence:** {confidence*100:.1f}%")
                        st.info("Try adjusting the confidence threshold or upload a clearer image.")
                        
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                with st.expander("Show error details"):
                    st.exception(e)
        else:
            st.info("👆 Upload an image to get started")

# TAB 2: MODEL PERFORMANCE
with tab2:
    st.subheader("📊 Model Evaluation Metrics")
    
    # Select model for metrics
    metrics_model = st.radio(
        "Select model to view metrics:",
        ["Custom CNN", "MobileNet"],
        horizontal=True
    )
    
    model_suffix = "cnn" if metrics_model == "Custom CNN" else "mobilenet"
    
    # Load metrics
    metrics_path = os.path.join(config.MODEL_DIR, f"metrics_{model_suffix}.npy")
    history_path = os.path.join(config.MODEL_DIR, f"history_{model_suffix}.npy")
    cm_path = os.path.join(config.MODEL_DIR, f"confusion_matrix_{model_suffix}.npy")
    
    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path, allow_pickle=True).item()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Test Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
        with col2:
            st.metric("Validation Accuracy", f"{metrics['val_accuracy']*100:.2f}%")
        with col3:
            st.metric("Training Time", f"{metrics['training_time_minutes']:.1f} min")
        with col4:
            st.metric("Test Loss", f"{metrics['test_loss']:.4f}")
        
        st.markdown("---")
        
        # Training history
        if os.path.exists(history_path):
            st.subheader("📈 Training History")
            history = np.load(history_path, allow_pickle=True).item()
            
            col1, col2 = st.columns(2)
            
            # Use Matplotlib to plot (best practice for Streamlit)
            with col1:
                # Accuracy plot
                fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
                ax_acc.plot(history['accuracy'], label='Train', linewidth=2)
                ax_acc.plot(history['val_accuracy'], label='Validation', linewidth=2)
                ax_acc.set_title('Model Accuracy', fontsize=14, fontweight='bold')
                ax_acc.set_xlabel('Epoch')
                ax_acc.set_ylabel('Accuracy')
                ax_acc.legend()
                ax_acc.grid(True, alpha=0.3)
                st.pyplot(fig_acc)
            
            with col2:
                # Loss plot
                fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
                ax_loss.plot(history['loss'], label='Train', linewidth=2)
                ax_loss.plot(history['val_loss'], label='Validation', linewidth=2)
                ax_loss.set_title('Model Loss', fontsize=14, fontweight='bold')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.legend()
                ax_loss.grid(True, alpha=0.3)
                st.pyplot(fig_loss)
        
        st.markdown("---")
        
        # Confusion Matrix
        if os.path.exists(cm_path):
            st.subheader("🔍 Confusion Matrix")
            
            cm = np.load(cm_path)
            
            # Plot confusion matrix
            fig_cm, ax_cm = plt.subplots(figsize=(14, 12))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax_cm,
                        cbar_kws={'label': 'Count'})
            ax_cm.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
            ax_cm.set_xlabel('Predicted Label', fontsize=12)
            ax_cm.set_ylabel('True Label', fontsize=12)
            st.pyplot(fig_cm)
            
            # Per-class accuracy
            st.subheader("📋 Per-Class Performance")
            class_accuracy = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏆 Best Performing Classes:**")
                # Ensure CLASS_NAMES is accessible, assumed to be in config
                best_indices = np.argsort(class_accuracy)[-5:][::-1]
                for idx in best_indices:
                    st.write(f"• Class {idx}: {config.CLASS_NAMES[idx][:35]} - **{class_accuracy[idx]*100:.1f}%**")
            
            with col2:
                st.markdown("**⚠️ Challenging Classes:**")
                worst_indices = np.argsort(class_accuracy)[:5]
                for idx in worst_indices:
                    st.write(f"• Class {idx}: {config.CLASS_NAMES[idx][:35]} - **{class_accuracy[idx]*100:.1f}%**")
        else:
            st.info("Confusion matrix data (.npy) not found.")
    else:
        st.warning(f"📭 No metrics found for {metrics_model}")
        st.info(f"""
        **Train the model first:**
        ```bash
        python src/train.py --model {model_suffix}
        ```
        """)

# TAB 3: DATA AUGMENTATION
with tab3:
    st.subheader("🎨 Data Augmentation Examples")
    
    st.markdown("""
    **Data augmentation improves model robustness by creating variations of training images.**
    
    Our augmentation techniques include:
    - 🔄 **Rotation**: ±15 degrees
    - ☀️ **Brightness**: 70-130%
    - 🔍 **Zoom**: 90-110%
    - ↔️ **Shifts**: ±10% horizontal/vertical
    """)
    
    if uploaded_file is not None and image_np is not None:
        st.markdown("---")
        st.markdown("### 🖼️ Augmented Versions of Your Image")
        
        # Ensure image_np is used (from Tab 1)
        img_original = image_np
        
        # Create augmentations
        cols = st.columns(4)
        
        # Original
        with cols[0]:
            st.image(img_original, caption="Original", use_column_width=True)
        
        # Rotated (simplified to 10 deg rotation for demonstration)
        with cols[1]:
            h, w = img_original.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 10, 1.0)
            img_rotated = cv2.warpAffine(img_original, M, (w, h))
            st.image(img_rotated, caption="Rotated +10°", use_column_width=True)
        
        # Brightened (alpha=1.3)
        with cols[2]:
            img_bright = cv2.convertScaleAbs(img_original, alpha=1.3, beta=0)
            st.image(img_bright, caption="Brightened (130%)", use_column_width=True)
        
        # Darkened (alpha=0.7)
        with cols[3]:
            img_dark = cv2.convertScaleAbs(img_original, alpha=0.7, beta=0)
            st.image(img_dark, caption="Darkened (70%)", use_column_width=True)
    else:
        st.info("👆 Upload an image in the Prediction tab to see augmentation examples")

# TAB 4: MODEL COMPARISON
with tab4:
    st.subheader("⚖️ Custom CNN vs MobileNet Comparison")
    
    st.markdown("""
    ### 🔬 Architecture Comparison
    
    | Feature | Custom CNN | MobileNet |
    |:---|:---|:---|
    | **Parameters** | ~500K | ~2M (frozen base) |
    | **Training** | From scratch | Transfer learning |
    | **Architecture** | 3 Conv blocks | Pre-trained ImageNet |
    | **Speed** | Fast inference | Moderate inference |
    | **Accuracy** | Good | Excellent |
    """)
    
    st.markdown("---")
    
    # Load both metrics if available
    cnn_metrics_path = os.path.join(config.MODEL_DIR, "metrics_cnn.npy")
    mobilenet_metrics_path = os.path.join(config.MODEL_DIR, "metrics_mobilenet.npy")
    
    if os.path.exists(cnn_metrics_path) and os.path.exists(mobilenet_metrics_path):
        cnn_metrics = np.load(cnn_metrics_path, allow_pickle=True).item()
        mobilenet_metrics = np.load(mobilenet_metrics_path, allow_pickle=True).item()
        
        st.subheader("📊 Performance Metrics Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 Custom CNN")
            st.metric("Test Accuracy", f"{cnn_metrics['test_accuracy']*100:.2f}%")
            st.metric("Training Time", f"{cnn_metrics['training_time_minutes']:.1f} min")
            st.metric("Test Loss", f"{cnn_metrics['test_loss']:.4f}")
        
        with col2:
            st.markdown("### 📱 MobileNet")
            st.metric("Test Accuracy", f"{mobilenet_metrics['test_accuracy']*100:.2f}%")
            st.metric("Training Time", f"{mobilenet_metrics['training_time_minutes']:.1f} min")
            st.metric("Test Loss", f"{mobilenet_metrics['test_loss']:.4f}")
        
        # Comparison chart
        st.markdown("---")
        st.subheader("📈 Visual Comparison")
        
        # Ensure plot generation is done correctly within Streamlit
        comparison_data = {
            'Custom CNN': [
                cnn_metrics['test_accuracy'] * 100,
                cnn_metrics['val_accuracy'] * 100,
                cnn_metrics['train_accuracy'] * 100
            ],
            'MobileNet': [
                mobilenet_metrics['test_accuracy'] * 100,
                mobilenet_metrics['val_accuracy'] * 100,
                mobilenet_metrics['train_accuracy'] * 100
            ]
        }
        
        fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
        x = np.arange(3)
        width = 0.35
        
        ax_comp.bar(x - width/2, comparison_data['Custom CNN'], width, label='Custom CNN', color='#FF6B6B')
        ax_comp.bar(x + width/2, comparison_data['MobileNet'], width, label='MobileNet', color='#4ECDC4')
        
        ax_comp.set_xlabel('Metrics', fontsize=12)
        ax_comp.set_ylabel('Accuracy (%)', fontsize=12)
        ax_comp.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels(['Test', 'Validation', 'Train'])
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig_comp)
        
        # Key findings
        st.markdown("---")
        st.subheader("🔑 Key Findings")
        
        better_model = "MobileNet" if mobilenet_metrics['test_accuracy'] > cnn_metrics['test_accuracy'] else "Custom CNN"
        faster_model = "Custom CNN" if cnn_metrics['training_time_minutes'] < mobilenet_metrics['training_time_minutes'] else "MobileNet"
        
        st.success(f"✅ **{better_model}** achieved higher test accuracy (Transfer Learning is often superior)")
        st.info(f"⚡ **{faster_model}** trained faster (fewer trainable parameters)")
        
    else:
        st.warning("📭 Train both models to see comparison")
        st.code("python src/train.py --model both")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Traffic Sign Recognition System</strong> | Built with Streamlit & TensorFlow</p>
    <p>GTSRB Dataset • 43 Classes • Deep Learning</p>
</div>
""", unsafe_allow_html=True)