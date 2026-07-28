import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="NutriGen AI",
    page_icon="🥗",
    layout="centered"
)

# -----------------------------
# Load AI Model (Only Once)
# -----------------------------
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🥗 NutriGen AI")
st.subheader("Personalized Nutrition Recommendation Engine")

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input("Age", 10, 100, 20)

weight = st.number_input(
    "Weight (kg)",
    min_value=20.0,
    max_value=200.0,
    value=60.0
)

height = st.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=250.0,
    value=170.0
)

# -----------------------------
# BMI Calculation
# -----------------------------
bmi = weight / ((height / 100) ** 2)

st.write(f"## BMI : {bmi:.2f}")

if bmi < 18.5:
    category = "Underweight"
elif bmi < 25:
    category = "Normal"
elif bmi < 30:
    category = "Overweight"
else:
    category = "Obese"
 
st.info(f"BMI Category : **{category}**")

st.markdown("---")

# -----------------------------
# Upload Food Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Food Image", 
    type=["jpg", "jpeg", "png","webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    with st.spinner("Detecting Food..."):
        prediction = model.predict(img_array)

    decoded = decode_predictions(prediction, top=1)[0]

    detected_food = decoded[0][1].replace("_", " ").lower()
    confidence = decoded[0][2] * 100

    st.success(f"Detected Food : {detected_food.title()}")
    st.write(f"Confidence : **{confidence:.2f}%**")

    # -----------------------------
    # Food Lists
    # -----------------------------
    healthy_foods = [
        "apple",
        "banana",
        "orange",
        "broccoli",
        "carrot",
        "cucumber",
        "spinach",
        "lettuce",
        "tomato",
        "grapes",
        "watermelon",
        "papaya",
        "egg",
        "fish",
        "chicken"
    ]

    junk_foods = [
        "pizza",
        "burger",
        "hotdog",
        "french fries",
        "donut",
        "cake",
        "cookie",
        "ice cream"
    ]

    # -----------------------------
    # Recommendation
    # -----------------------------
    if category == "Underweight":
        recommendation = "Eat nutritious high-calorie foods."

    elif category == "Normal":
        recommendation = "Balanced diet recommended."

    elif category == "Overweight":
        if detected_food in junk_foods:
            recommendation = "Limit Intake"
        else:
            recommendation = "Healthy Choice"

    else:
        if detected_food in junk_foods:
            recommendation = "Avoid"
        else:
            recommendation = "Healthy Choice"

    st.subheader("Recommendation")
    st.success(recommendation)

    # -----------------------------
    # Suggested Foods
    # -----------------------------
    st.subheader("Recommended Foods")

    if category == "Underweight":
        st.write("""
        ✅ Milk  
        ✅ Eggs  
        ✅ Rice  
        ✅ Banana  
        ✅ Fish  
        ✅ Nuts
        """)

    elif category == "Normal":
        st.write("""
        ✅ Apple  
        ✅ Chicken  
        ✅ Brown Rice  
        ✅ Salad  
        ✅ Fruits  
        ✅ Vegetables
        """)

    elif category == "Overweight":
        st.write("""
        ✅ Apple  
        ✅ Broccoli  
        ✅ Cucumber  
        ✅ Salad  
        ✅ Green Tea
        """)

    else:
        st.write("""
        ✅ Broccoli  
        ✅ Spinach  
        ✅ Apple  
        ✅ Cucumber  
        ✅ Water  
        ✅ Oats
        """)

    # -----------------------------
    # Nutrition Facts
    # -----------------------------
    nutrition = {
        "apple": {
            "Calories": "52 kcal",
            "Protein": "0.3 g",
            "Carbohydrates": "14 g",
            "Fat": "0.2 g"
        },
        "banana": {
            "Calories": "89 kcal",
            "Protein": "1.1 g",
            "Carbohydrates": "23 g",
            "Fat": "0.3 g"
        },
        "orange": {
            "Calories": "47 kcal",
            "Protein": "0.9 g",
            "Carbohydrates": "12 g",
            "Fat": "0.1 g"
        },
        "broccoli": {
            "Calories": "34 kcal",
            "Protein": "2.8 g",
            "Carbohydrates": "7 g",
            "Fat": "0.4 g"
        }
    }

    if detected_food in nutrition:
        st.markdown("---")
        st.subheader("Nutrition Facts")

        for key, value in nutrition[detected_food].items():
            st.write(f"**{key}:** {value}")

st.markdown("---")
st.caption("NutriGen AI | Powered by TensorFlow & Streamlit")