# FitAI - AI-Based Virtual Shopping Assistant 👗🤖

FitAI is a virtual shopping assistant built with Python and Streamlit that uses Artificial Intelligence (AI) to recommend clothing based on a user's body shape, skin tone, and gender. Just upload a photo, and FitAI does the rest!

##  Features

- 🔍 **Pose Detection** using MediaPipe
- 🧠 **Body Shape Classification** using Neural Network & Random Forest
- 👤 **Gender Prediction** using a CNN (VGG16-based)
- 🎨 **Skin Tone Analysis** using facial region color analysis
- 👕 **AI-based Clothing Recommendations**
- 🛒 **Google Shopping Integration**
- 💾 **User Data Storage** with SQLite
- 📈 **Synthetic Data Generation** for model training
- 🧪 Train & load models automatically if not available

##  Tech Stack

- Python
- Streamlit
- TensorFlow (Neural Networks)
- scikit-learn (Random Forest, Label Encoding)
- MediaPipe (Pose Estimation, Face Mesh)
- OpenCV (Face Detection)
- SQLite (User history storage)
- VGG16 CNN (Gender prediction)

## Model Files

- `body_shape_nn_model.h5` – Neural network for body shape
- `body_shape_rf_model.pkl` – Random forest for body shape
- `clothing_recommendation_model.pkl` – Clothing suggestion model
- `gender_model.h5` – Gender prediction model

##  How It Works

1. Upload your image (selfie or full-body).
2. AI detects your gender, body shape, and skin tone.
3. Based on analysis, FitAI suggests personalized clothing.
4. Click shopping links to buy recommended outfits!

##  User History

- See previously analyzed results using the "User History" tab.

##  Developed By

Aesthetic Era✨
