# 🧠 Brain Tumor Detection System

A deep learning-based application that detects brain tumors from MRI images using a Convolutional Neural Network (CNN). This project includes an end-to-end pipeline — from dataset preprocessing and model training to deployment via a web interface.

---

## 📌 Overview

Brain tumors are critical medical conditions that require early detection. This project leverages deep learning techniques to automate tumor detection from MRI scans, assisting in faster and more consistent analysis.

The system is designed as a **binary image classification model**:
- **Class 0** → No Tumor  
- **Class 1** → Tumor Present  

---

## 🏗️ System Architecture

```
MRI Image → Preprocessing → CNN Model → Prediction → Streamlit UI
```

---

## 🧠 Model Details

### Architecture
- Convolutional Neural Network (CNN)
- Layers:
  - Conv2D + Batch Normalization + ReLU
  - MaxPooling layers
  - Fully connected Dense layers
  - Dropout (to reduce overfitting)

### Configuration
| Parameter       | Value                  |
|----------------|------------------------|
| Input Size      | 128 × 128 × 3         |
| Loss Function   | Binary Crossentropy    |
| Optimizer       | Adam                   |
| Regularization  | Dropout, Batch Norm, Data Augmentation |

---

## 📊 Performance Metrics

| Metric     | Value |
|-----------|-------|
| Accuracy  | 93%   |
| Precision | 0.93  |
| Recall    | 0.93  |
| F1 Score  | 0.93  |

### Confusion Matrix
```
[[387  28]
 [ 27 309]]
```
- **False Positives:** 28  
- **False Negatives:** 27  

The model demonstrates strong generalization with balanced performance across both classes.

---

## 📁 Project Structure

```
Brain-Tumor-Detector/
│
├── Datasets/
│   ├── yes/                  # Tumor images
│   └── no/                   # Non-tumor images
│
├── brain_tumor_model.h5      # Trained model
├── mainTrain.py              # Training pipeline
├── app.py                    # Streamlit application
├── requirements.txt          # Dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/Ojaswin30/Brain-Tumor-Detector.git
cd Brain-Tumor-Detector
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / Mac
.venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Model Training

Run the training pipeline:
```bash
python mainTrain.py
```

This will:
- Load and preprocess the dataset
- Apply data augmentation
- Train the CNN model
- Save the trained model as `brain_tumor_model.h5`

---

## 🌐 Running the Application

Start the Streamlit server:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Then open the provided URL in your browser.

---

## 📸 Application Workflow

1. Upload an MRI image
2. Image is preprocessed (resize + normalization)
3. Model performs prediction
4. Result is displayed:
   - ✅ **No Tumor Detected**
   - ⚠️ **Tumor Detected**

---

## 📦 Dependencies

- TensorFlow / Keras
- NumPy
- OpenCV
- Pillow
- Streamlit
- scikit-learn

---

## ⚠️ Limitations

- Model trained on a limited dataset (~3,700 images)
- May not generalize to all MRI scan variations
- False negatives still exist (~8%)
- Not suitable for clinical deployment

---

## 🚀 Future Enhancements

- Implement Transfer Learning (e.g., MobileNetV2)
- Add Grad-CAM for visual explainability
- Improve recall to reduce false negatives
- Deploy as a scalable REST API
- Integrate with frontend frameworks (React / Power Apps)

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. It should not be used for medical diagnosis. Always consult certified medical professionals for clinical decisions.

---

## 👨‍💻 Author

**Ojaswin Aggarwal**  
[GitHub](https://github.com/Ojaswin30)

---

## ⭐ Contributions

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or raise an issue.
