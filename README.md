# 🚦 Traffic Sign Recognition

A Deep Learning project for classifying 43 types of German traffic signs using Convolutional Neural Networks (CNNs) and Transfer Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- ✅ **3 Different Model Architectures** - Simple CNN, Deep CNN, and MobileNet
- ✅ **Interactive Web Interface** - Built with Streamlit
- ✅ **Comprehensive Evaluation** - Accuracy, Precision, Recall, F1-Score
- ✅ **Model Comparison Dashboard** - Compare all models side-by-side
- ✅ **Real-time Predictions** - Upload and classify traffic signs instantly
- ✅ **Image Format Support** - Handles PNG (with transparency), JPG, JPEG
- ✅ **Detailed Analytics** - Confusion matrix analysis and per-class accuracy

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ RAM (4GB recommended)

### Installation

**Step 1: Clone the repository**

```bash
git clone <your-repo-url>
cd traffic-sign-recognition
```

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3: Prepare your dataset**

Download the GTSRB dataset and organize it as follows:

```
data/GTSRB/
├── Train.csv
├── Test.csv
├── Meta.csv
├── Train/        # Folder containing training images
└── Test/         # Folder containing test images
```

**Step 4: Train the models**

```bash
python train_all_models.py
```

⏱️ Expected time: 15-25 minutes (depending on hardware)

**Step 5: Launch the web app**

```bash
streamlit run app.py
```

🌐 The app will open at `http://localhost:8501`

## 📊 Model Performance

| Model          | Test Accuracy | Precision | Recall | F1-Score | Inference Speed | Parameters |
| -------------- | ------------- | --------- | ------ | -------- | --------------- | ---------- |
| **Simple CNN** | 97.21%        | 97.31%    | 97.21% | 97.20%   | 0.62 ms         | 549,355    |
| **Deep CNN**   | 98.96%        | 98.99%    | 98.96% | 98.96%   | 0.87 ms         | 677,323    |
| **MobileNet**  | 32.31%\*      | 30.56%    | 32.31% | 29.99%   | 0.66 ms         | 2,427,499  |

_Note: MobileNet requires further tuning for better performance on 32×32 images_

## 🏗️ Project Structure

```
traffic-sign-recognition/
│
├── data/
│   └── GTSRB/              # GTSRB dataset
│       ├── Train.csv
│       ├── Test.csv
│       ├── Meta.csv
│       ├── Train/          # Training images
│       └── Test/           # Test images
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── models.py           # CNN architectures
│   ├── train.py            # Training functions
│   └── evaluate.py         # Evaluation metrics
│
├── saved_models/           # Trained model files (.h5)
│   ├── simple_cnn.h5
│   ├── deep_cnn.h5
│   └── mobilenet.h5
│
├── results/
│   └── evaluation_results.json  # Model metrics
│
├── config.py               # Configuration settings
├── train_all_models.py     # Training pipeline script
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎨 Web Application

### Tab 1: 🔍 Predict Sign

- **Upload Image**: Drag & drop or browse for traffic sign images
- **Model Selection**: Choose between Simple CNN, Deep CNN, or MobileNet
- **Results Display**:
  - Top prediction with confidence score
  - Top 5 predictions with progress bars
  - Downloadable results as text file

### Tab 2: 📊 Model Comparison

- **Performance Charts**: Interactive bar charts and radar plots
- **Metrics Comparison**: Accuracy, Precision, Recall, F1-Score
- **Speed Analysis**: Inference time comparison
- **Parameter Count**: Model complexity comparison

### Tab 3: 📈 Detailed Evaluation

- **Confusion Matrix**: Interactive table with color gradients
- **Statistics**: Correct predictions, worst confusions, best class
- **Correct Predictions Chart**: Bar chart showing diagonal values
- **Top 10 Confused Pairs**: Detailed table of misclassifications
- **Per-Class Accuracy**: Best and worst performing traffic signs
- **Accuracy Distribution**: Bar chart for all 43 classes

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Image settings
IMG_SIZE = (32, 32)          # Image dimensions
NUM_CLASSES = 43             # Number of traffic sign classes

# Training settings
BATCH_SIZE = 64              # Training batch size
EPOCHS = 30                  # Maximum training epochs
LEARNING_RATE = 0.001        # Learning rate
VALIDATION_SPLIT = 0.2       # Validation data percentage

# Paths
DATA_DIR = "data/GTSRB"
MODEL_DIR = "saved_models"
RESULTS_DIR = "results"
```

## 🤖 Model Architectures

### Simple CNN

- **Architecture**: 2 convolutional blocks
- **Strengths**: Fast training, low memory usage
- **Best for**: Quick prototyping, resource-constrained environments

### Deep CNN

- **Architecture**: 4 convolutional blocks with Batch Normalization
- **Strengths**: Highest accuracy, robust performance
- **Best for**: Production deployment, best results

### MobileNet (Transfer Learning)

- **Architecture**: Pre-trained on ImageNet, custom classification head
- **Strengths**: Efficient architecture, transfer learning benefits
- **Best for**: Mobile/edge deployment (after proper fine-tuning)

## 📚 Dataset Information

**GTSRB - German Traffic Sign Recognition Benchmark**

- **Source**: [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_news.html)
- **Classes**: 43 different traffic sign types
- **Training Images**: ~39,000
- **Test Images**: ~12,630
- **Image Size**: Variable (resized to 32×32 for this project)
- **Format**: PPM/PNG

**Traffic Sign Categories**:

- Speed limits (20-120 km/h)
- Prohibitory signs (No entry, No passing, etc.)
- Mandatory signs (Turn right/left, Roundabout, etc.)
- Warning signs (Curves, Pedestrians, Animals, etc.)

## 🐛 Troubleshooting

### Models not loading

```bash
# Train models first
python train_all_models.py
```

### Out of memory error

```python
# In config.py, reduce:
BATCH_SIZE = 32  # or 16
```

### Slow training

```python
# In config.py, reduce:
EPOCHS = 30
```

### Import errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### RGBA image error

✅ **Already handled!** The code automatically converts RGBA images to RGB.

## 📦 Dependencies

```
tensorflow==2.15.0      # Deep learning framework
streamlit==1.29.0       # Web application framework
pandas==2.1.4           # Data manipulation
numpy==1.24.3           # Numerical computing
Pillow==10.1.0          # Image processing
scikit-learn==1.3.2     # Machine learning utilities
plotly==5.18.0          # Interactive visualizations
```

## 🎓 Use Cases

- **Autonomous Vehicles**: Real-time traffic sign detection
- **Driver Assistance Systems**: Alert drivers about road signs
- **Traffic Management**: Automated sign recognition for monitoring
- **Educational**: Learning computer vision and deep learning
- **Research**: Benchmark for image classification algorithms

## 🚧 Known Limitations

- **Training Data**: Only German traffic signs (GTSRB dataset)
- **Image Size**: Optimized for 32×32 images
- **MobileNet Performance**: Requires fine-tuning for small images
- **Real-world Variability**: Performance may vary with different lighting, angles, or countries

## 🔮 Future Enhancements

- [ ] Data augmentation for better generalization
- [ ] Test-Time Augmentation (TTA)
- [ ] Support for international traffic signs
- [ ] Real-time video detection
- [ ] Model quantization for edge deployment
- [ ] REST API for model serving
- [ ] Docker containerization

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark)
- **Framework**: TensorFlow/Keras team
- **UI Framework**: Streamlit community
- **Inspiration**: Autonomous driving research

## 📧 Contact

For questions or feedback:

- Create an issue in this repository
- Email: [tharusharasatml@gmail.com]
- LinkedIn: [[Your LinkedIn Profile](https://www.linkedin.com/in/tharusha-rasath-5b9643243/)]

**⭐ If you found this project helpful, please give it a star!**
