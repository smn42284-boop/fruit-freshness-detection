# 🍎 Fresh vs Rotten Fruit Detection

A deep learning system that classifies fruits and vegetables as fresh or rotten using Convolutional Neural Networks (CNNs).

## 🎯 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **96.29%** |
| AUC Score | **0.992** |
| Precision (Fresh) | 96.78% |
| Recall (Rotten) | 97.05% |

## 🏗️ Model Architecture

- 4 Convolutional layers (32, 64, 128, 128 filters)
- MaxPooling after each conv layer
- Dense layer with 512 neurons
- Dropout (0.5) for regularization
- Output: Sigmoid for binary classification

## 📊 Key Features

- Custom CNN trained from scratch (no pre-trained weights)
- Handles 6 fruit/vegetable types
- Real-time web interface with Streamlit
- 100% accuracy on real-world kitchen tests

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run demo_app.py
