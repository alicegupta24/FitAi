# Import necessary libraries
import streamlit as st

# Set page config (must be the first Streamlit command)
st.set_page_config(
    page_title="FitAI - Virtual Shopping Assistant",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

import cv2
import numpy as np
import mediapipe as mp
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import joblib
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
from contextlib import contextmanager
import gdown
import requests
import zipfile
import io
import pandas as pd
from urllib.parse import quote_plus


# Load Haar Cascade model for face detection using OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Simulated dataset for training ML models (replace with actual dataset in production)
X_sample = np.random.rand(100, 66)  # 33 pose landmarks with x and y => 66 features
y_sample = np.random.choice([0, 1, 2], 100)  # Sample class labels

# Train a simple neural network model for body shape prediction
def train_nn_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(66,)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_sample, y_sample, epochs=10, verbose=0)
    model.save("body_shape_nn_model.h5")
    return model

# Train a Random Forest model for body shape prediction
def train_rf_model():
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_sample, y_sample)
    joblib.dump(rf, "body_shape_rf_model.pkl")
    return rf

# Train recommendation model for clothing based on gender, body shape, and skin tone
def train_recommendation_model():
    data = [
    ('Male', 'Triangle', 'Warm', 'V-neck shirts, warm earth tones'),
    ('Male', 'Triangle', 'Cool', 'Straight-cut jeans, navy & grey'),
    ('Male', 'Triangle', 'Neutral', 'Beige chinos and neutral t-shirts'),
    ('Male', 'Rectangle', 'Warm', 'Layered shirts, khaki pants'),
    ('Male', 'Rectangle', 'Cool', 'Slim-fit blazers, cool-toned shirts'),
    ('Male', 'Rectangle', 'Neutral', 'White tees, grey pants'),
    ('Male', 'Oval', 'Warm', 'Vertical stripe shirts, warm trousers'),
    ('Male', 'Oval', 'Cool', 'Dark jackets, blue denim'),
    ('Male', 'Oval', 'Neutral', 'Comfort fit polos, black jeans'),
    ('Male', 'Diamond', 'Warm', 'Casual open shirts, orange tones'),
    ('Male', 'Diamond', 'Cool', 'Relaxed navy sweatshirts, jeans'),
    ('Male', 'Diamond', 'Neutral', 'Relaxed-fit shirts, neutral colors'),
    ('Female', 'Hourglass', 'Warm', 'Fitted dresses, warm-colored outfits'),
    ('Female', 'Hourglass', 'Cool', 'A-line dresses, blue & purple tones'),
    ('Female', 'Hourglass', 'Neutral', 'Peplum tops, balanced tones'),
    ('Female', 'Pear', 'Warm', 'Bright tops, earthy skirts'),
    ('Female', 'Pear', 'Cool', 'Flowy tunics, cool trousers'),
    ('Female', 'Pear', 'Neutral', 'Neutral blouses, black leggings'),
    ('Female', 'Apple', 'Warm', 'Empire waist dresses, coral tones'),
    ('Female', 'Apple', 'Cool', 'V-neck tunics, cool pastels'),
    ('Female', 'Apple', 'Neutral', 'Wrap tops, neutral tones'),
    ('Female', 'Hourglass', 'Warm', 'Bodycon outfits in orange/red tones'),
    ('Female', 'Hourglass', 'Cool', 'Cool blue tailored dresses'),
    ('Female', 'Hourglass', 'Neutral', 'Soft beige tops with pencil skirts'),
    ('Female', 'Pear', 'Warm', 'Tops with detailing, tan slacks'),
    ('Female', 'Pear', 'Cool', 'Printed tops, dark bottoms'),
    ('Female', 'Pear', 'Neutral', 'White shirts, navy skirts'),
    ('Female', 'Apple', 'Warm', 'Loose blouses, red accessories'),
    ('Female', 'Apple', 'Cool', 'Structured jackets, light blues'),
    ('Female', 'Apple', 'Neutral', 'Cream tunics with black jeans'),
    ('Male', 'Triangle', 'Warm', 'Rust polos, cargo pants'),
    ('Male', 'Triangle', 'Cool', 'Blue plaid shirts, dark jeans'),
    ('Male', 'Triangle', 'Neutral', 'Grey sweaters, denim'),
    ('Male', 'Rectangle', 'Warm', 'Tan cardigans, olive pants'),
    ('Male', 'Rectangle', 'Cool', 'Grey jackets, teal t-shirts'),
    ('Male', 'Rectangle', 'Neutral', 'Simple shirts, jeans'),
    ('Male', 'Oval', 'Warm', 'Maroon sweaters, beige trousers'),
    ('Male', 'Oval', 'Cool', 'Navy coats, white shirts'),
    ('Male', 'Oval', 'Neutral', 'Graphite hoodies, neutral chinos'),
    ('Male', 'Diamond', 'Warm', 'Open linen shirts, sand tones'),
    ('Male', 'Diamond', 'Cool', 'Soft fleece pullovers, blue jeans'),
    ('Male', 'Diamond', 'Neutral', 'Relaxed blazers, off-white trousers'),
    ('Female', 'Hourglass', 'Warm', 'Warm fitted crop tops, maxi skirts'),
    ('Female', 'Hourglass', 'Cool', 'Cool ruffle tops, pencil skirts'),
    ('Female', 'Hourglass', 'Neutral', 'Balanced tones with tailored dresses'),
    ('Female', 'Pear', 'Warm', 'Ruffled blouses, tan capris'),
    ('Female', 'Pear', 'Cool', 'Grey sweaters, navy leggings'),
    ('Female', 'Pear', 'Neutral', 'Neutral wrap tops, bootcut jeans'),
    ('Female', 'Apple', 'Warm', 'Vibrant scarves, loose blouses'),
    ('Female', 'Apple', 'Cool', 'Soft jackets, pastel trousers'),
    ('Female', 'Apple', 'Neutral', 'Balanced shirts with dark pants'),
    ('Male', 'Triangle', 'Cool', 'Ocean blue shirts, black joggers'),
    ('Male', 'Triangle', 'Warm', 'Mustard jackets, brown pants'),
    ('Male', 'Triangle', 'Neutral', 'Plain white tees, navy trousers'),
    ('Male', 'Rectangle', 'Cool', 'Steel blue sweaters, black jeans'),
    ('Male', 'Rectangle', 'Warm', 'Camel coats, olive chinos'),
    ('Male', 'Rectangle', 'Neutral', 'Heather shirts, neutral khakis'),
    ('Male', 'Oval', 'Cool', 'Blue bombers, white jeans'),
    ('Male', 'Oval', 'Warm', 'Amber jackets, dark jeans'),
    ('Male', 'Oval', 'Neutral', 'Off-white hoodies, jeans'),
    ('Female', 'Hourglass', 'Cool', 'Navy blouses, tailored pants'),
    ('Female', 'Hourglass', 'Warm', 'Rust midi dresses'),
    ('Female', 'Hourglass', 'Neutral', 'Gray pencil skirts and tops'),
    ('Female', 'Pear', 'Cool', 'Light cardigans, dark jeggings'),
    ('Female', 'Pear', 'Warm', 'Terracotta wrap tops, beige pants'),
    ('Female', 'Pear', 'Neutral', 'White tops, olive jeans'),
    ('Female', 'Apple', 'Cool', 'Structured pastel blazers'),
    ('Female', 'Apple', 'Warm', 'Sunset-tone dresses'),
    ('Female', 'Apple', 'Neutral', 'Neutral tunics, straight pants'),
    ('Male', 'Diamond', 'Cool', 'Dark denim with grey pullovers'),
    ('Male', 'Diamond', 'Warm', 'Khaki shorts, coral polos'),
    ('Male', 'Diamond', 'Neutral', 'Light jackets, earth-toned pants'),
    ('Female', 'Diamond', 'Cool', 'Grey jumpers, navy skirts'),
    ('Female', 'Diamond', 'Warm', 'Sunset shades in loose fits'),
    ('Female', 'Diamond', 'Neutral', 'Cream blazers with tan pants'),
]
    df = np.array(data)
    X = df[:, :3]  # Features: Gender, Body Shape, Skin Tone
    y = df[:, 3]   # Target: Clothing recommendation

    # Label encode categorical features
    le_gender = LabelEncoder()
    le_body_shape = LabelEncoder()
    le_skin_tone = LabelEncoder()
    le_clothing = LabelEncoder()

    le_gender.fit(X[:, 0])
    le_body_shape.fit(X[:, 1])
    le_skin_tone.fit(X[:, 2])
    le_clothing.fit(y)

    X_encoded = np.column_stack([
        le_gender.transform(X[:, 0]),
        le_body_shape.transform(X[:, 1]),
        le_skin_tone.transform(X[:, 2])
    ])
    y_encoded = le_clothing.transform(y)

    rf_recommendation = RandomForestClassifier(n_estimators=100)
    rf_recommendation.fit(X_encoded, y_encoded)

    # Save model and encoders
    joblib.dump((rf_recommendation, le_gender, le_body_shape, le_skin_tone, le_clothing), "clothing_recommendation_model.pkl")
    return rf_recommendation, le_gender, le_body_shape, le_skin_tone, le_clothing

# Shopping integration functions
def get_shopping_links(clothing_item, gender):
    """Get Google Shopping link for recommended clothing items with gender specification"""
    search_query = f"{gender}'s {clothing_item}"
    return f"https://www.google.com/search?q={quote_plus(search_query)}&tbm=shop"

def display_shopping_links(recommended_clothes, gender):
    """Display Google Shopping links for recommended items with gender specification"""
    # Split the recommendation into individual items
    items = [item.strip() for item in recommended_clothes.split(',')]
    
    st.write("**Shop These Looks:**")
    for item in items:
        if item:  # Check if item is not empty
            google_shopping_link = get_shopping_links(item, gender)
            st.markdown(f"[Shop for {gender}'s {item}]({google_shopping_link})", unsafe_allow_html=True)

# Load or train models (caching for performance)
@st.cache_resource
def load_models():
    nn = tf.keras.models.load_model("body_shape_nn_model.h5") if os.path.exists("body_shape_nn_model.h5") else train_nn_model()
    rf = joblib.load("body_shape_rf_model.pkl") if os.path.exists("body_shape_rf_model.pkl") else train_rf_model()
    if os.path.exists("clothing_recommendation_model.pkl"):
        return (nn, rf, *joblib.load("clothing_recommendation_model.pkl"))
    else:
        return (nn, rf, *train_recommendation_model())

# Load all models
nn_model, rf_model, recommendation_model, le_gender, le_body_shape, le_skin_tone, le_clothing = load_models()

# Initialize MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Initialize MediaPipe Face Mesh for better facial feature detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Database connection management
@contextmanager
def get_database_connection():
    conn = None
    try:
        conn = sqlite3.connect("users.db", check_same_thread=False)
        yield conn
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# Initialize database
def init_database():
    with get_database_connection() as conn:
        try:
            cursor = conn.cursor()
            
            # Create table if it doesn't exist (don't drop existing table)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    gender TEXT NOT NULL,
                    body_shape TEXT,
                    skin_tone TEXT,
                    recommended_clothes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error initializing database: {e}")

# Save user data to database
def save_user_data(user_data):
    with get_database_connection() as conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (name, gender, body_shape, skin_tone, recommended_clothes)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_data['name'],
                user_data['gender'],
                user_data['body_shape'],
                user_data['skin_tone'],
                user_data['recommended_clothes']
            ))
            conn.commit()
            return True
        except sqlite3.Error as e:
            st.error(f"Error saving user data: {e}")
            return False

# Get all user data
def get_all_users():
    with get_database_connection() as conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, gender, body_shape, skin_tone, recommended_clothes, created_at FROM users ORDER BY created_at DESC")
            columns = ['ID', 'Name', 'Gender', 'Body Shape', 'Skin Tone', 'Recommended Clothes', 'Created At']
            rows = cursor.fetchall()
            return columns, rows
        except sqlite3.Error as e:
            st.error(f"Error retrieving user data: {e}")
            return None, None

# Initialize database at app startup
init_database()

# Detect if a face is present in the image
def contains_face(image):
    image = image.convert("RGB")
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return len(faces) > 0

# Extract 33 keypoints from the image using MediaPipe Pose
def extract_keypoints(image):
    image = image.convert("RGB")
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = pose.process(img_array)
    if results.pose_landmarks:
        keypoints = [val for lm in results.pose_landmarks.landmark for val in (lm.x, lm.y)]
        return np.array(keypoints).reshape(1, -1)
    return None

# Predict body shape using the neural network
def predict_body_shape(image, gender):
    keypoints = extract_keypoints(image)
    if keypoints is not None and keypoints.shape[1] == 66:
        predicted_label_nn = np.argmax(nn_model.predict(keypoints, verbose=0))
        body_shapes = {
            "Male": ["Triangle", "Rectangle", "Oval", "Diamond"],
            "Female": ["Hourglass", "Pear", "Apple", "Diamond"]
        }
        return body_shapes[gender][predicted_label_nn] if predicted_label_nn < len(body_shapes[gender]) else "Unknown"
    return "Unknown"

# Determine skin tone from the center of the face
def get_skin_tone(image):
    image = image.convert("RGB")
    img = np.array(image)
    h, w, _ = img.shape
    cx, cy = w // 2, h // 3  # Approximate forehead region
    region = img[cy-20:cy+20, cx-20:cx+20]
    avg_color = np.mean(region.reshape(-1, 3), axis=0)
    r, g, b = avg_color
    if r > g and r > b:
        return "Warm"
    elif b > r and b > g:
        return "Cool"
    else:
        return "Neutral"

# Recommend clothing based on predicted features
def recommend_clothes(body_shape, skin_tone, gender):
    try:
        input_data = np.array([
            le_gender.transform([gender])[0],
            le_body_shape.transform([body_shape])[0],
            le_skin_tone.transform([skin_tone])[0]
        ]).reshape(1, -1)
        recommended_index = recommendation_model.predict(input_data)[0]
        return le_clothing.inverse_transform([recommended_index])[0]
    except:
        return "No recommendation available."

# Create gender classification model using VGG16
def create_gender_classifier():
    # Load VGG16 model without top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create new model
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

# Function to download and prepare the dataset
def download_and_prepare_dataset():
    dataset_path = 'gender_dataset'
    model_path = 'gender_model.h5'
    
    # If model already exists, return
    if os.path.exists(model_path):
        return model_path
        
    # Create dataset directory if it doesn't exist
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        
        # Download UTKFace dataset (smaller subset)
        st.info("Downloading dataset... This may take a few minutes.")
        url = "https://drive.google.com/uc?id=1ZtEhVPXqjqPA-6V4me8Omh_4Uas4Af3R"
        
        try:
            response = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(dataset_path)
            st.success("Dataset downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading dataset: {str(e)}")
            return None
    
    # Create and train the model
    model = create_and_train_model()
    model.save(model_path)
    return model_path

# Function to create and train the model
def create_and_train_model():
    dataset_path = 'gender_dataset'
    model_path = 'gender_model.h5'
    
    # If model already exists, load it
    if os.path.exists(model_path):
        return load_model(model_path)
    
    # Create CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Load and prepare dataset
    X_train = []
    y_train = []
    
    # Define dataset folders
    dataset_folders = ['crop_part1', 'UTKFace', 'utkface_aligned_cropped']
    total_images = 0
    
    # Count total images first
    for folder in dataset_folders:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            total_images += len(files)
    
    # Progress bar for dataset loading
    progress_bar = st.progress(0)
    status_text = st.empty()
    processed_images = 0
    
    # Process images from each folder
    for folder in dataset_folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            continue
            
        st.info(f"Loading images from {folder}...")
        
        file_list = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_name in file_list:
            try:
                # Update progress
                processed_images += 1
                progress = processed_images / total_images
                progress_bar.progress(progress)
                status_text.text(f"Loading images... {processed_images}/{total_images}")
                
                # Extract gender from filename (UTKFace format: [age]_[gender]_[race]_[date&time].jpg)
                gender = int(img_name.split('_')[1])
                
                # Load and preprocess image
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.resize(img, (64, 64))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                
                X_train.append(img)
                y_train.append(gender)
                
            except Exception as e:
                continue
    
    # Clear progress bar and status
    progress_bar.empty()
    status_text.empty()
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    if len(X_train) == 0:
        st.error("No valid images found in the dataset directories!")
        return None
    
    # Display dataset info
    st.info(f"Dataset loaded: {len(X_train)} images")
    
    # Train the model with progress bar
    st.info("Training the model... This may take a few minutes.")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save the trained model
    model.save(model_path)
    st.success("Model trained and saved successfully!")
    
    # Display training results
    val_accuracy = history.history['val_accuracy'][-1]
    st.info(f"Validation accuracy: {val_accuracy:.2%}")
    
    return model

# Initialize gender classification model
@st.cache_resource
def get_gender_classifier():
    return create_and_train_model()

# Preprocess image for gender classification
def preprocess_image_for_gender(image):
    try:
        # Convert PIL Image to cv2 format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Use the first detected face
            face_img = img[y:y+h, x:x+w]
            
            # Add padding around the face
            padding = int(w * 0.3)  # 30% padding
            start_y = max(0, y - padding)
            end_y = min(img.shape[0], y + h + padding)
            start_x = max(0, x - padding)
            end_x = min(img.shape[1], x + w + padding)
            face_img = img[start_y:end_y, start_x:end_x]
            
            # Resize and preprocess
            face_img = cv2.resize(face_img, (64, 64))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = face_img / 255.0
            return np.expand_dims(face_img, axis=0)
        return None
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Predict gender
def predict_gender(image):
    try:
        model = get_gender_classifier()
        if model is None:
            st.error("Could not load or train gender classification model")
            return None, None
            
        preprocessed_image = preprocess_image_for_gender(image)
        if preprocessed_image is not None:
            prediction = model.predict(preprocessed_image)[0][0]
            gender = "Male" if prediction < 0.5 else "Female"  # UTKFace: 0 for male, 1 for female
            confidence = 1 - prediction if prediction < 0.5 else prediction
            return gender, float(confidence)
        return None, None
    except Exception as e:
        st.error(f"Error in gender prediction: {str(e)}")
        return None, None

# Streamlit App UI
st.title("FitAI: AI-Based Virtual Shopping Assistant")

# Add sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Home", "User History"])

if page == "Home":
    st.markdown("""
    Upload a **clear selfie or full-body image**.  
    The assistant analyzes your **gender**, **body shape**, **skin tone**, and provides **clothing suggestions**.
    """)

    # User inputs
    user_name = st.text_input("Enter your name:")
    uploaded_file = st.file_uploader("Upload a selfie or photo:", type=["jpg", "png", "jpeg"])

    # Run analysis after input and upload
    if uploaded_file and user_name:
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, caption="Uploaded Image", width=600)

        if not contains_face(image):
            st.error("No face detected. Please upload a clearer selfie or image with a visible face.")
        else:
            # Predict gender
            gender, confidence = predict_gender(image)
            if gender and confidence:
                st.write(f"✅ Detected Gender: **{gender}** (Confidence: {confidence:.2%})")
                
                keypoints = extract_keypoints(image)
                if keypoints is not None:
                    st.write("✅ Keypoints detected successfully")
                    
                    body_shape = predict_body_shape(image, gender)
                    st.write("✅ Body shape prediction complete")

                    if body_shape == "Unknown":
                        st.warning("Could not identify body shape clearly. Try uploading a clearer full-body image.")
                    else:
                        skin_tone = get_skin_tone(image)
                        recommended_clothes = recommend_clothes(body_shape, skin_tone, gender)

                        # Display results
                        st.subheader("Analysis Results")
                        st.write(f"**Body Shape:** {body_shape}")
                        st.write(f"**Skin Tone:** {skin_tone}")
                        st.write(f"**Recommended Clothing:** {recommended_clothes}")
                        
                        # Add shopping links section
                        st.markdown("---")
                        display_shopping_links(recommended_clothes, gender)

                        # Save to database
                        user_data = {
                            'name': user_name,
                            'gender': gender,
                            'body_shape': body_shape,
                            'skin_tone': skin_tone,
                            'recommended_clothes': recommended_clothes
                        }
                        
                        if save_user_data(user_data):
                            st.success("Your results have been saved successfully!")
                        else:
                            st.error("There was an error saving your results.")
                else:
                    st.error("Could not detect pose landmarks. Please upload a clearer full-body image.")
            else:
                st.error("Could not determine gender from the image. Please upload a clearer face image.")

elif page == "User History":
    st.title("User History")
    columns, rows = get_all_users()
    if columns and rows:
        for row in rows:
            with st.expander(f"{row[1]}"):
                st.write(f"**Analysis Date:** {row[6]}")
                st.markdown("---")
                st.write(f"**Gender:** {row[2]}")
                st.write(f"**Body Shape:** {row[3]}")
                st.write(f"**Skin Tone:** {row[4]}")
                st.write(f"**Recommended Clothes:**")
                st.write(f"{row[5]}")
    else:
        st.info("No previous analyses yet")

# Add footer with credits
st.markdown("""
<div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: white; padding: 10px; text-align: center;">
    <p style="color: black; margin: 0;">Developed by Aesthetic Era And Team</p>
</div>
""", unsafe_allow_html=True)
