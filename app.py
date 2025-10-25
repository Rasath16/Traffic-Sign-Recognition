import streamlit as st
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.evaluate import load_evaluation_results
from src.data_loader import preprocess_single_image
import config as config


# Page config
st.set_page_config(
    page_title=" Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #667eea;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .model-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: #667eea;
        color: white;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/traffic-light.png", width=100)
    st.title("Navigation")
    st.markdown("---")
    
    st.markdown("###  Quick Stats")
    results = load_evaluation_results()
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        st.success(f"**Best Model:** {best_model[0]}")
        st.info(f"**Accuracy:** {best_model[1]['accuracy']:.2f}%")
    
    st.markdown("---")
    st.markdown("###  Features")
    st.markdown("✅ Real-time prediction")
    st.markdown("✅ 3 Model comparison")
    st.markdown("✅ Detailed analytics")
    st.markdown("✅ 43 Sign classes")
    
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("[ Documentation](#)")
    st.markdown("[ GitHub Repo](#)")
    st.markdown("[ Report Bug](#)")

@st.cache_resource
def load_all_models():
    models = {}
    try:
        models['Simple CNN'] = load_model(config.SIMPLE_CNN_PATH)
        models['Deep CNN'] = load_model(config.DEEP_CNN_PATH)
        models['MobileNet'] = load_model(config.MOBILENET_PATH)
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

def main():
    # Animated header
    st.markdown('<p class="main-header">🚦 Traffic Sign Recognition</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Deep Learning & Computer Vision</p>', 
                unsafe_allow_html=True)
    
    # Load resources
    models = load_all_models()
    results = load_evaluation_results()
    
    if not models or not results:
        st.error("⚠️ Models not trained yet!")
        st.info("Run: `python train_all_models.py`")
        return
    
    # Tabs with icons
    tab1, tab2, tab3, = st.tabs([
        "🔍 Predict", 
        "📊 Compare", 
        "📈 Evaluate",
        
    ])
    
    with tab1:
        prediction_tab_enhanced(models, results)
    
    with tab2:
        comparison_tab_enhanced(results)
    
    with tab3:
        evaluation_tab_enhanced(results)
    
    
def prediction_tab_enhanced(models, results):
    """Enhanced prediction tab with better UI"""
    st.header(" Upload & Predict Traffic Sign")
    
    # Model selection at top
    col_model, col_info = st.columns([2, 1])
    
    with col_model:
        model_choice = st.selectbox(
            "🤖 Choose AI Model:",
            list(models.keys()),
            help="Select which model to use for prediction"
        )
    
    with col_info:
        if model_choice in results:
            acc = results[model_choice]['accuracy']
            st.metric("Model Accuracy", f"{acc:.2f}%", 
                     delta=f"{acc - 85:.1f}% vs baseline")
    
    st.markdown("---")
    
    # Upload section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(" Upload Image")
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a traffic sign"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='📸 Your Image', use_container_width=True)
            
            # Image info
            st.info(f"📐 Size: {image.size[0]} x {image.size[1]} pixels")
    
    with col2:
        st.subheader(" Prediction Results")
        
        if uploaded_file:
            predict_btn = st.button("🚀 Analyze Traffic Sign", 
                                   use_container_width=True,
                                   type="primary")
            
            if predict_btn:
                with st.spinner("🧠 AI is analyzing your image..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                    
                    # Predict
                    img_array = preprocess_single_image(uploaded_file)
                    model = models[model_choice]
                    predictions = model.predict(img_array, verbose=0)[0]
                    
                    # Get top prediction
                    top_idx = np.argmax(predictions)
                    confidence = predictions[top_idx] * 100
                    
                    # Result box
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🎯 Detected Sign</h2>
                        <h1>{config.CLASS_NAMES[top_idx]}</h1>
                        <h3>Confidence: {confidence:.2f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top 5 predictions with progress bars
                    st.subheader("📊 Top 5 Predictions")
                    top_5_idx = np.argsort(predictions)[-5:][::-1]
                    
                    for i, idx in enumerate(top_5_idx):
                        conf = predictions[idx] * 100
                        if i == 0:
                            st.success(f"🥇 {config.CLASS_NAMES[idx]}: {conf:.2f}%")
                        elif i == 1:
                            st.info(f"🥈 {config.CLASS_NAMES[idx]}: {conf:.2f}%")
                        elif i == 2:
                            st.warning(f"🥉 {config.CLASS_NAMES[idx]}: {conf:.2f}%")
                        else:
                            st.write(f"{i+1}. {config.CLASS_NAMES[idx]}: {conf:.2f}%")
                        
                        st.progress(float(predictions[idx]))
                    
                    # Download results
                    st.markdown("---")
                    result_text = f"""
                    Traffic Sign Recognition Results
                    ================================
                    Model: {model_choice}
                    Predicted Sign: {config.CLASS_NAMES[top_idx]}
                    Confidence: {confidence:.2f}%
                    
                    Top 5 Predictions:
                    """
                    for i, idx in enumerate(top_5_idx):
                        result_text += f"\n{i+1}. {config.CLASS_NAMES[idx]}: {predictions[idx]*100:.2f}%"
                    
                    st.download_button(
                        "📥 Download Results",
                        result_text,
                        "prediction_results.txt",
                        use_container_width=True
                    )
        else:
            st.info("👆 Upload an image to get started!")

def comparison_tab_enhanced(results):
    """Enhanced comparison with better visualizations"""
    st.header("🏆 Model Performance Showdown")
    
    # Quick comparison cards
    cols = st.columns(3)
    for idx, (model_name, model_results) in enumerate(results.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{model_name}</h3>
                <h1>{model_results['accuracy']:.2f}%</h1>
                <p>Accuracy</p>
                <hr>
                <p>F1: {model_results['f1_score']:.2f}%</p>
                <p>Speed: {model_results['avg_inference_per_image']:.2f}ms</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive comparison
    st.subheader("📊 Interactive Metrics Comparison")
    
    # Metric selector
    metric_choice = st.selectbox(
        "Select Metric to Compare:",
        ['accuracy', 'precision', 'recall', 'f1_score', 'avg_inference_per_image'],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Create comparison chart
    model_names = list(results.keys())
    values = [results[model][metric_choice] for model in model_names]
    
    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=values,
            text=[f'{v:.2f}' for v in values],
            textposition='auto',
            marker=dict(
                color=values,
                colorscale='Viridis',
                showscale=True
            )
        )
    ])
    
    fig.update_layout(
        title=f"{metric_choice.replace('_', ' ').title()} Comparison",
        yaxis_title="Score" if 'time' not in metric_choice else "Time (ms)",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart for multiple metrics
    st.subheader("🎯 Multi-Metric Radar Chart")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig = go.Figure()
    
    for model_name in model_names:
        values = [results[model_name][m] for m in metrics]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[m.title() for m in metrics],
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("📋 Complete Metrics Table")
    
    comparison_data = []
    for model_name, model_results in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{model_results['accuracy']:.2f}%",
            'Precision': f"{model_results['precision']:.2f}%",
            'Recall': f"{model_results['recall']:.2f}%",
            'F1-Score': f"{model_results['f1_score']:.2f}%",
            'Parameters': f"{model_results['total_params']:,}",
            'Speed (ms)': f"{model_results['avg_inference_per_image']:.2f}",
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Highlight best values
    st.dataframe(
        df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                              color='lightgreen')
               .highlight_min(subset=['Speed (ms)'], color='lightblue'),
        use_container_width=True
    )

def evaluation_tab_enhanced(results):
    """Enhanced evaluation with more insights"""
    st.header("📈 Deep Dive: Model Evaluation")
    
    # Model selector
    selected_model = st.selectbox(
        "🔍 Select Model for Analysis:",
        list(results.keys())
    )
    
    model_results = results[selected_model]
    
    # Key metrics
    st.subheader("📊 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", f"{model_results['accuracy']:.2f}%")
    with col2:
        st.metric("⚡ Speed", f"{model_results['avg_inference_per_image']:.2f}ms")
    with col3:
        st.metric("🧠 Params", f"{model_results['total_params']:,}")
    with col4:
        st.metric("📊 F1-Score", f"{model_results['f1_score']:.2f}%")
    
    st.markdown("---")
    
    # Confusion Matrix - AS INTERACTIVE TABLE
    st.subheader("🔥 Confusion Matrix")
    
    cm = np.array(model_results['confusion_matrix'])
    
    # Create DataFrame with nice labels
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {i}" for i in range(config.NUM_CLASSES)],
        columns=[f"Pred {i}" for i in range(config.NUM_CLASSES)]
    )
    
    # Show with color gradient
    st.dataframe(
        cm_df.style.background_gradient(
            cmap='YlOrRd',
            axis=None,
            vmin=0,
            vmax=cm.max()
        ).set_properties(**{
            'font-size': '10px',
            'text-align': 'center'
        }).format("{:.0f}"),
        height=600,
        use_container_width=True
    )
    
    st.caption("📊 **How to read:** Yellow = low counts | Orange = medium | Red = high counts | Diagonal should be darkest (correct predictions)")
    

    
    # Per-class accuracy
    st.markdown("---")
    st.subheader("📊 Per-Class Performance Analysis")
    
    class_acc_dict = model_results['per_class_accuracy']
    sorted_classes = sorted(class_acc_dict.items(), key=lambda x: x[1], reverse=True)
    best_5 = sorted_classes[:5]
    worst_5 = sorted_classes[-5:]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("🏆 Top 5 Best Performing Classes")
        for sign, acc in best_5:
            st.write(f"**{sign}**: {acc:.2f}%")
            st.progress(min(acc/100, 1.0))
    
    with col2:
        st.error("⚠️ Top 5 Challenging Classes")
        for sign, acc in worst_5:
            st.write(f"**{sign}**: {acc:.2f}%")
            st.progress(min(acc/100, 1.0))
    
    # Full distribution
    st.markdown("---")
    st.subheader("📈 All Classes Accuracy Distribution")
    
    classes = list(class_acc_dict.keys())
    accuracies = [class_acc_dict[c] for c in classes]
    
    fig2 = go.Figure(data=[
        go.Bar(
            x=list(range(len(classes))),
            y=accuracies,
            marker=dict(
                color=accuracies,
                colorscale='RdYlGn',
                cmin=0,
                cmax=100,
                showscale=True,
                colorbar=dict(title="Accuracy %")
            ),
            hovertext=[f"{c}<br>{a:.1f}%" for c, a in zip(classes, accuracies)],
            hoverinfo='text'
        )
    ])
    
    fig2.update_layout(
        xaxis_title="Class Index",
        yaxis_title="Accuracy (%)",
        height=400,
        yaxis=dict(range=[0, 105])
    )
    
    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()