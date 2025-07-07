import streamlit as st
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
import io
import datetime

# --------------------------
# Load Classifier & Feature Extractor
# --------------------------
@st.cache_resource
def load_model():
    model_path = "ResNet50_Logistic_Regression.pkl"
    classifier = joblib.load(model_path)
    feature_extractor = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return classifier, feature_extractor

classifier, feature_extractor = load_model()

# --------------------------
# Class Mapping & Treatment Suggestions
# --------------------------
class_indices = {
    'Anthracnose'    : 0,
    'Bacterial_Blight': 1,
    'Citrus_Canker'  : 2,
    'Curl_Virus'     : 3,
    'Deficiency_Leaf': 4,
    'Dry_Leaf'       : 5,
    'Healthy_Leaf'   : 6,
    'Sooty_Mould'    : 7,
    'Spider_Mites'   : 8
}
inverse_class_indices = {v: k for k, v in class_indices.items()}

treatment_suggestions = {
    'Anthracnose': (
        "Anthracnose is a fungal disease causing dark, sunken lesions on leaves.\n"
        "- Prune and destroy infected leaves and twigs to reduce the spread.\n"
        "- Apply copper-based fungicides during wet seasons, especially after heavy rainfall.\n"
        "- Ensure proper air circulation by spacing plants and avoiding overhead irrigation."
    ),
    'Bacterial_Blight': (
        "Bacterial blight causes leaf spots, blights, and eventual defoliation.\n"
        "- Remove and burn infected leaves to stop further spread.\n"
        "- Spray bactericides like streptomycin or copper oxychloride weekly.\n"
        "- Avoid watering late in the day and prevent leaf wetness overnight.\n"
        "- Rotate crops and use disease-free planting material."
    ),
    'Citrus_Canker': (
        "Citrus canker causes raised lesions on leaves, stems, and fruits.\n"
        "- Remove and destroy infected leaves and branches immediately.\n"
        "- Regularly apply copper-based bactericides to protect healthy tissues.\n"
        "- Quarantine new plants and disinfect pruning tools.\n"
        "- Consider windbreaks to reduce spread by wind-driven rain."
    ),
    'Curl_Virus': (
        "Curl virus causes leaf curling and stunted growth.\n"
        "- There is no cure once a plant is infected.\n"
        "- Control insect vectors like aphids using insecticidal soaps or neem oil.\n"
        "- Remove and destroy severely infected plants.\n"
        "- Use certified virus-free seeds and planting material."
    ),
    'Deficiency_Leaf': (
        "Nutrient deficiencies lead to discoloration or distortion of leaves.\n"
        "- Conduct a soil test to identify missing nutrients.\n"
        "- Apply specific fertilizers: e.g., zinc sulfate for zinc deficiency, or Epsom salts for magnesium.\n"
        "- Use organic compost to improve soil health and nutrient availability.\n"
        "- Maintain proper pH to enhance nutrient uptake (ideal: 6.0–6.5 for citrus)."
    ),
    'Dry_Leaf': (
        "Dry leaf symptoms may indicate dehydration or salt buildup.\n"
        "- Water deeply and regularly, especially during dry spells.\n"
        "- Mulch around the base to retain soil moisture.\n"
        "- Avoid over-fertilization, which can cause leaf burn.\n"
        "- Improve soil drainage if waterlogging occurs."
    ),
    'Healthy_Leaf': (
        "The leaf appears healthy and free from visible disease.\n"
        "- Continue regular monitoring for pests or changes in leaf color.\n"
        "- Maintain balanced watering and feeding schedules.\n"
        "- Periodically inspect plants for early signs of disease or insect activity."
    ),
    'Sooty_Mould': (
        "Sooty mold is a black fungus that grows on honeydew excreted by insects.\n"
        "- First, control sap-sucking insects like aphids, mealybugs, or whiteflies.\n"
        "- Use neem oil or insecticidal soap sprays to eliminate insect sources.\n"
        "- Gently wash the leaves with warm water and mild soap to remove mold.\n"
        "- Encourage natural predators like ladybugs in your garden."
    ),
    'Spider_Mites': (
        "Spider mites cause yellow speckling and webbing on leaves.\n"
        "- Increase humidity around plants, as mites thrive in dry conditions.\n"
        "- Spray plants with a strong stream of water to knock mites off.\n"
        "- Use miticides or horticultural oils (e.g., neem, canola oil).\n"
        "- Introduce beneficial mites or insects (like lacewings) for biological control."
    )
}

# --------------------------
# Streamlit UI
# --------------------------
st.title("🍋 Lemon Leaf Disease Classifier")
st.write("Upload an image of a lemon leaf to predict the disease, view treatment advice, and download a diagnosis report.")

uploaded_file = st.file_uploader("📤 Upload a lemon leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    img = image_display.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features
    features = feature_extractor.predict(img_array)
    features_flattened = features.reshape(1, -1)

    # Predict with probabilities
    prediction_probs = classifier.predict_proba(features_flattened)[0]
    predicted_class_idx = np.argmax(prediction_probs)
    predicted_class = inverse_class_indices[int(predicted_class_idx)]
    confidence = prediction_probs[predicted_class_idx] * 100

    # Display result
    st.success(f"🩺 Predicted Disease: **{predicted_class}**")
    st.info(f"🔍 Confidence: **{confidence:.2f}%**")
    st.markdown(f"💊 **Treatment Suggestion:** {treatment_suggestions[predicted_class]}")

    # Downloadable report
    report_text = f"""
Lemon Leaf Disease Diagnosis Report
===================================
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Prediction: {predicted_class}
Confidence: {confidence:.2f}%
Treatment: {treatment_suggestions[predicted_class]}

Thank you for using the Lemon Disease Classifier.
"""

    # Convert to bytes
    buffer = io.BytesIO()
    buffer.write(report_text.encode())
    buffer.seek(0)

    st.download_button(
        label="📄 Download Diagnosis Report",
        data=buffer,
        file_name="lemon_disease_report.txt",
        mime="text/plain"
    )
