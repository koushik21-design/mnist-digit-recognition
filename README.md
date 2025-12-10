# Handwritten Digit Recognition (MNIST)
This is a **Machine Learning + Deep Learning project** that recognizes handwritten digits (0–9) using
a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.
The app is built with **Streamlit**, allowing users to:
- ✏️ Draw digits directly in the browser canvas
- Upload single-digit images
- Upload images containing multiple handwritten digits
…and get instant predictions!
---
## Project Overview
### Objective
To identify handwritten digits from user input using a CNN model trained on the MNIST dataset.
### Technologies Used
- **Python**
- **TensorFlow / Keras**
- **Streamlit**
- **Pillow (PIL)**
- **NumPy & Pandas**
- **streamlit-drawable-canvas**
---
## How It Works
1. **Model Training**
 - A CNN is trained on MNIST dataset (60,000 training, 10,000 testing samples).
 - Model achieves around **98–99% accuracy**.
 - The trained model is saved as `mnist_cnn.h5`.
2. **Web App**
 - Built using Streamlit.
 - Loads the trained model.
 - Supports three input modes:
 - Draw a digit
 - Upload a single digit image
 - Upload an image with multiple digits
 - Automatically preprocesses input to match MNIST format (28×28 grayscale).
3. **Prediction**
 - The model outputs probabilities for digits 0–9.
 - The highest probability determines the predicted digit.
---
## Features
| Feature | Description |
|----------|--------------|
| Draw Digit | Draw any digit 0–9 directly on screen |
| Upload Image | Upload a single digit photo for recognition |
| Multi-digit Support | Upload an image containing multiple digits (like “2025”) |
| Probabilities | See confidence scores for each digit |
| Real-time Processing | Instant predictions through Streamlit interface |
---
## How to Run Locally
### 1 Clone the repository
``bash
git clone https://github.com/koushik21-design/mnist-digit-recognition.git
cd mnist-digit-recognition
### 2 Create a virtual environment
``bash
python -m venv venv
venv\Scripts\activate
### 3 Install dependencies
``bash
pip install -r requirements.txt
### 4️ Run the Streamlit app
``bash
streamlit run app.py


 
 Dependencies:
streamlit
tensorflow
Pillow
numpy
pandas
streamlit-drawable-canvas



 Author:
Yeruva Koushik Reddy
 B.Tech in CSE (AI & ML), R.V.R & J.C College of Engineering, Guntur
 [koushikreddyyeruva21@gmail.com]
