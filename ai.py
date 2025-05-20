import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import os
import random

st.set_page_config(page_title="AI-Based Virtual Shopping Assistant", layout="centered")
st.title("🛍️ AI-Based Virtual Shopping Assistant")

# --- Model Training Improvements ---
def generate_synthetic_data(num_samples=300):
    X = []
    y = []
    for i in range(num_samples):
        gender = np.random.choice([0, 1])  # 0=Male, 1=Female
        body_shape = np.random.choice(4)
        base = np.random.rand(66) * 0.1
        pattern = np.zeros(66)
        pattern[body_shape*16:(body_shape+1)*16] += 0.3
        sample = base + pattern
        X.append(sample)
        y.append(body_shape)
    return np.array(X), np.array(y)

def train_nn_model():
    X_train, y_train = generate_synthetic_data()
    model = Sequential([
        Dense(128, activation='relu', input_shape=(66,)),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, verbose=0)
    model.save("body_shape_nn_model.h5")
    return model

def train_rf_model():
    X_train, y_train = generate_synthetic_data()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "body_shape_rf_model.pkl")
    return rf

@st.cache_resource
def load_models():
    nn = tf.keras.models.load_model("body_shape_nn_model.h5") if os.path.exists("body_shape_nn_model.h5") else train_nn_model()
    rf = joblib.load("body_shape_rf_model.pkl") if os.path.exists("body_shape_rf_model.pkl") else train_rf_model()
    return nn, rf

nn_model, rf_model = load_models()

# --- Recommendation Data ---
RECOMMENDATION_DATA = [
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

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# --- OpenCV Haar Cascade for Face Detection ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Database Setup ---
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        gender TEXT,
        body_shape TEXT,
        skin_tone TEXT,
        recommended_clothes TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

try:
    cursor.execute("ALTER TABLE users ADD COLUMN timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")
    conn.commit()
except sqlite3.OperationalError:
    pass

# --- Face Detection with fallback ---
def contains_face(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) > 0:
        return True
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = face_detection.process(img_bgr)
    if results.detections:
        return True
    return False

# --- Keypoints Extraction ---
def extract_keypoints(image):
    img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    results = pose.process(img)
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
        return np.array(keypoints).reshape(1, -1)
    return None

# --- Predict Body Shape ---
def predict_body_shape(image, gender):
    keypoints = extract_keypoints(image)
    if keypoints is not None and keypoints.shape[1] == 66:
        prediction_probs = nn_model.predict(keypoints, verbose=0)
        predicted_label_nn = np.argmax(prediction_probs)
        body_shapes = {
            "Male": ["Triangle", "Rectangle", "Oval", "Diamond"],
            "Female": ["Hourglass", "Pear", "Apple", "Diamond"]
        }
        return body_shapes.get(gender, ["Unknown"]*4)[predicted_label_nn]
    return "Unknown"

# --- Improved Skin Tone Detection ---
def get_skin_tone(image):
    img = np.array(image.convert("RGB"))
    h, w, _ = img.shape
    cx, cy = w // 2, h // 3
    region = img[max(cy-20,0):cy+20, max(cx-20,0):cx+20]
    region_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    avg_hue = np.mean(region_hsv[:, :, 0])
    if avg_hue < 30 or avg_hue > 150:
        return "Warm"
    elif 90 <= avg_hue <= 150:
        return "Cool"
    else:
        return "Neutral"

# --- Random Clothing Recommendation ---
def recommend_clothes(body_shape, skin_tone, gender):
    matches = [rec for (g, b, s, rec) in RECOMMENDATION_DATA if g == gender and b == body_shape and s == skin_tone]
    if matches:
        return random.choice(matches)
    else:
        return "No recommendation available."

# --- Streamlit UI ---
menu = st.sidebar.selectbox("Menu", ["Home", "Saved Analyses", "Clear Database"])

if menu == "Home":
    st.markdown("""
    Upload a **clear selfie or full-body image**.  
    The assistant analyzes your **body shape**, **skin tone**, and provides **clothing suggestions**.
    """)

    user_name = st.text_input("👤 Enter your name:")
    gender = st.radio("🧑‍🤝‍🧑 Select your gender:", ("Male", "Female"))
    uploaded_file = st.file_uploader("📷 Upload a selfie or photo:", type=["jpg", "png", "jpeg"])

    if uploaded_file and user_name:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if not contains_face(image):
            st.error("❌ No face detected. Please upload a clearer selfie or image with a visible face.")
        else:
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    keypoints = extract_keypoints(image)
                    st.write("✅ Keypoints detected" if keypoints is not None else "⚠️ Pose landmarks not detected")

                    body_shape = predict_body_shape(image, gender)
                    st.write("🔍 Body shape prediction complete.")

                    if body_shape == "Unknown":
                        st.warning("⚠️ Could not identify body shape clearly. Try uploading a clearer full-body image.")
                    else:
                        skin_tone = get_skin_tone(image)
                        recommended_clothes = recommend_clothes(body_shape, skin_tone, gender)

                        st.subheader("📊 Analysis Results")
                        st.write(f"**👗 Body Shape:** {body_shape}")
                        st.write(f"**🎨 Skin Tone:** {skin_tone}")
                        st.write(f"**👚 Clothing Recommendation:** {recommended_clothes}")

                        # Save to DB
                        cursor.execute("""
                            INSERT INTO users (name, gender, body_shape, skin_tone, recommended_clothes)
                            VALUES (?, ?, ?, ?, ?)
                        """, (user_name, gender, body_shape, skin_tone, recommended_clothes))
                        conn.commit()
                        st.success("✅ Analysis saved successfully.")

elif menu == "Saved Analyses":
    st.header("📂 Saved User Analyses")
    search_name = st.text_input("Search by user name:")
    query = "SELECT name, gender, body_shape, skin_tone, recommended_clothes, timestamp FROM users"
    params = ()
    if search_name:
        query += " WHERE name LIKE ?"
        params = ('%' + search_name + '%',)
    query += " ORDER BY timestamp DESC LIMIT 100"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            st.markdown(f"""
            **Name:** {row[0]}  
            **Gender:** {row[1]}  
            **Body Shape:** {row[2]}  
            **Skin Tone:** {row[3]}  
            **Clothing Recommendation:** {row[4]}  
            **Timestamp:** {row[5]}  
            ---
            """)
    else:
        st.info("No saved analyses found.")

elif menu == "Clear Database":
    st.warning("⚠️ This will delete all saved analyses. This action cannot be undone.")
    if st.button("Clear All"):
        cursor.execute("DELETE FROM users")
        conn.commit()
        st.success("All analyses cleared.")
