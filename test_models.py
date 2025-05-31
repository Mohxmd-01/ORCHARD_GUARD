import joblib
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
import cv2 as cv

# Load the models
svm_model = joblib.load('apple_leaf_svm_model.joblib')
label_encoder = joblib.load('apple_leaf_label_encoder.joblib')

# Print model information
print("\nModel Information:")
print("SVM Model classes:", svm_model.classes_)
print("Label Encoder classes:", label_encoder.classes_)

# Create a simple feature extractor (same as in the app)
base_model = EfficientNetB0(weights='imagenet', include_top=False,
                           input_shape=(224, 224, 3),
                           pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output,
                         name="EfficientNetB0_Feature_Extractor")

# Test with a random image (we'll use one from the dataset)
import os
categories = ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"]

def test_prediction(image_path):
    try:
        # Load and preprocess image
        img = cv.imread(image_path)
        img_resized = cv.resize(img, (224, 224))
        img_for_nn = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_for_nn.copy())
        
        # Extract features
        features = feature_extractor.predict(img_preprocessed, verbose=0)
        
        # Make prediction
        prediction = svm_model.predict(features)
        probabilities = svm_model.predict_proba(features)
        
        # Get predicted class and probabilities
        predicted_class = categories[prediction[0]]
        prob_dict = {categories[i]: float(probabilities[0][i]) for i in range(len(categories))}
        
        print("\nTest Prediction Results:")
        print(f"Predicted class: {predicted_class}")
        print("\nProbabilities:")
        for cls, prob in prob_dict.items():
            print(f"{cls}: {prob*100:.2f}%")
            
    except Exception as e:
        print(f"Error during test prediction: {e}")

# Find a test image
for category in categories:
    category_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), category)
    if os.path.exists(category_path) and os.listdir(category_path):
        test_image = os.path.join(category_path, os.listdir(category_path)[0])
        print(f"\nTesting with image from {category} category:")
        test_prediction(test_image)
        break
