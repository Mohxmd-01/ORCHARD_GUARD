import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
# from sklearn.preprocessing import LabelEncoder # Not strictly needed for prediction if CATEGORIES list is used
import joblib # For loading SVM model and label encoder
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS # Import CORS
from werkzeug.utils import secure_filename
import traceback # For more detailed error logging

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allows frontend from different origins

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Folder to temporarily store uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model and Global Variables ---
# These will be loaded once when the app starts
FEATURE_EXTRACTOR = None
SVM_MODEL = None
# LABEL_ENCODER = None # Loaded for verification, but string mapping uses CATEGORIES
IMAGE_SIZE = (224, 224)  # Must match the training image size

# CATEGORIES: This list is CRUCIAL. It defines the mapping from the numeric output
# of the SVM model (0, 1, 2, 3) to human-readable string names.
# The order MUST EXACTLY MATCH the order used during training in train_model.py
# (i.e., the order in the CATEGORIES list in train_model.py).
# Default hardcoded categories, can be overridden by dynamic loading if class_names_from_training.txt exists.
CATEGORIES = ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"] 

MODEL_PATH_SVM = "apple_leaf_svm_model.joblib"
MODEL_PATH_LE = "apple_leaf_label_encoder.joblib" # Path to the saved LabelEncoder
CLASS_NAMES_FILE_PATH = "class_names_from_training.txt" # Path to the file with class names

def configure_tensorflow_gpu():
    """Configures TensorFlow to use GPU if available and sets memory growth."""
    print("Configuring TensorFlow for GPU...")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s):")
            for gpu_device in gpus: # Renamed to avoid conflict
                print(f"  {gpu_device.name}")
                try:
                    tf.config.experimental.set_memory_growth(gpu_device, True)
                    print(f"    Memory growth enabled for {gpu_device.name}")
                except RuntimeError as e:
                    print(f"    Error setting memory growth for {gpu_device.name}: {e} (Must be set before GPUs have been initialized if already in use by another process)")
        else:
            print("No GPU found by TensorFlow. Using CPU.")
    except Exception as e:
        print(f"Error during TensorFlow GPU configuration: {e}")

def load_application_models():
    """Loads the feature extractor, SVM model, label encoder, and dynamically CATEGORIES if file exists."""
    global FEATURE_EXTRACTOR, SVM_MODEL, CATEGORIES # LABEL_ENCODER is loaded but not essential for prediction logic here
    label_encoder_for_verification = None # Local variable for the loaded LabelEncoder
    print("Loading application models...")
    try:
        # --- Dynamically load CATEGORIES from file if it exists ---
        # This makes the app more robust if train_model.py changes the order/names.
        if os.path.exists(CLASS_NAMES_FILE_PATH):
            loaded_categories_from_file = []
            with open(CLASS_NAMES_FILE_PATH, "r") as f:
                for line in f:
                    try:
                        # Assuming format "index:ClassName" or just "ClassName" per line
                        parts = line.strip().split(":", 1)
                        name = parts[1] if len(parts) > 1 else parts[0]
                        if name: # Ensure name is not empty
                            loaded_categories_from_file.append(name)
                    except IndexError:
                        print(f"Warning: Malformed line in {CLASS_NAMES_FILE_PATH}: {line.strip()}")
            if loaded_categories_from_file:
                CATEGORIES = loaded_categories_from_file
                print(f"Dynamically loaded CATEGORIES from {CLASS_NAMES_FILE_PATH}: {CATEGORIES}")
            else:
                print(f"Warning: {CLASS_NAMES_FILE_PATH} was empty or malformed. Using hardcoded CATEGORIES: {CATEGORIES}")
        else:
            print(f"Warning: {CLASS_NAMES_FILE_PATH} not found. Using hardcoded CATEGORIES: {CATEGORIES}")
        # --- End Dynamic Load ---

        # 1. Load EfficientNetB0 Feature Extractor
        base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                    pooling='avg')
        FEATURE_EXTRACTOR = Model(inputs=base_model.input, outputs=base_model.output,
                                  name="EfficientNetB0_Feature_Extractor_App")
        print("EfficientNetB0 feature extractor loaded.")

        # 2. Load SVM Model
        if not os.path.exists(MODEL_PATH_SVM):
            raise FileNotFoundError(f"SVM model not found at {MODEL_PATH_SVM}. Run train_model.py first.")
        SVM_MODEL = joblib.load(MODEL_PATH_SVM)
        print(f"SVM model loaded. SVM internal classes (should be numeric, e.g. [0 1 2 3]): {SVM_MODEL.classes_}")

        # 3. Load Label Encoder (primarily for verification of consistency with training)
        if not os.path.exists(MODEL_PATH_LE):
            raise FileNotFoundError(f"Label encoder not found at {MODEL_PATH_LE}. Run train_model.py first.")
        label_encoder_for_verification = joblib.load(MODEL_PATH_LE)
        print(f"Label encoder loaded (for verification). Encoder internal classes (should be numeric, e.g. [0 1 2 3]): {label_encoder_for_verification.classes_}")

        # Crucial Check: Ensure SVM model's learned classes count matches CATEGORIES list length.
        if not (len(SVM_MODEL.classes_) == len(CATEGORIES)):
            print("CRITICAL WARNING: Mismatch between SVM model's number of classes and CATEGORIES list length!")
            print(f"  SVM model's number of classes: {len(SVM_MODEL.classes_)}, CATEGORIES length: {len(CATEGORIES)}")
            print("  This will lead to incorrect class name mapping during prediction.")
            return False # Prevent app from starting with this critical mismatch

        print("All models loaded successfully.")
        return True
    except FileNotFoundError as fnf_error:
        print(f"MODEL LOADING ERROR: {fnf_error}")
        print("Please ensure 'train_model.py' has been run successfully and model files are in the correct location.")
    except Exception as e:
        print(f"An unexpected error occurred while loading models: {e}")
        traceback.print_exc()
    return False


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_leaf_disease_from_path(image_path):
    """
    Predicts disease from an image file path.
    Uses globally loaded models and the global CATEGORIES list for string names.
    """
    if not all([FEATURE_EXTRACTOR, SVM_MODEL]):
        return "Error: Core models (Feature Extractor or SVM) not loaded.", None

    try:
        img_array = cv2.imread(image_path)
        if img_array is None:
            return f"Error: Could not read image at {image_path}.", None

        img_resized = cv2.resize(img_array, IMAGE_SIZE)
        img_for_nn = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_for_nn.copy())

        # Extract features using EfficientNet
        features = FEATURE_EXTRACTOR.predict(img_preprocessed, verbose=0) # verbose=0 for less console output
        
        # Predict numeric class index using SVM (e.g., 0, 1, 2, or 3)
        prediction_numeric_index = SVM_MODEL.predict(features)[0]
        # Get probabilities for each class from SVM
        prediction_proba_svm = SVM_MODEL.predict_proba(features) 
        
        # Map numeric index to string class name using the global CATEGORIES list
        predicted_class_name_str = "Unknown"
        if 0 <= prediction_numeric_index < len(CATEGORIES):
            predicted_class_name_str = CATEGORIES[prediction_numeric_index]
        else:
            print(f"WARNING: Predicted numeric index {prediction_numeric_index} is out of bounds for CATEGORIES list (len: {len(CATEGORIES)}).")
            predicted_class_name_str = f"Unknown Class (Raw Index: {prediction_numeric_index})"

        # Create dictionary of probabilities with string class names as keys
        probabilities_dict = {}
        if len(prediction_proba_svm[0]) == len(CATEGORIES):
            for i, class_name_key in enumerate(CATEGORIES):
                probabilities_dict[class_name_key] = float(prediction_proba_svm[0][i])
        else:
            print(f"CRITICAL WARNING in prediction: Length of SVM probability output ({len(prediction_proba_svm[0])}) "
                  f"does not match length of CATEGORIES ({len(CATEGORIES)}). Probabilities will be misaligned.")
            for i in range(len(prediction_proba_svm[0])): # Fallback if lengths mismatch
                key_name = CATEGORIES[i] if i < len(CATEGORIES) else f"Internal_Class_Index_{i}"
                probabilities_dict[key_name] = float(prediction_proba_svm[0][i])
        
        return predicted_class_name_str, probabilities_dict

    except Exception as e:
        print(f"Error during prediction for image {image_path}:")
        traceback.print_exc()
        return f"Error during prediction: {str(e)}", None


@app.route('/', methods=['GET'])
def home_page():
    """Serves the main analyze.html page or a simple status message."""
    # Assumes analyze.html is your main interaction page and is in the 'templates' folder
    if os.path.exists(os.path.join(app.template_folder, 'analyze.html')):
         return render_template('analyze.html')
    # Fallback if analyze.html is not found in templates
    return "Leaf Disease Analysis API is running. Ensure 'analyze.html' is in the 'templates' folder or use the /api/predict endpoint."


@app.route('/api/predict', methods=['POST'])
def api_predict_leaf_disease():
    """Handles image upload via API and returns JSON prediction."""
    if 'leaf_image' not in request.files:
        return jsonify({"error": "No image file provided in the 'leaf_image' field.", "status": "error"}), 400
    
    file = request.files['leaf_image']
    
    if file.filename == '':
        return jsonify({"error": "No image selected for uploading.", "status": "error"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(temp_image_path)
            print(f"API: Image saved temporarily to: {temp_image_path}")

            predicted_class, probabilities = predict_leaf_disease_from_path(temp_image_path)

            if probabilities: # Check if probabilities dict is not None
                prediction_data = {
                    "image_filename": filename,
                    "predicted_class": predicted_class,
                    "probabilities": probabilities,
                    "status": "success"
                }
                return jsonify(prediction_data), 200
            else:
                # predicted_class here contains the error message from predict_leaf_disease_from_path
                return jsonify({"error": predicted_class, "status": "error"}), 500 
        except Exception as e:
            print(f"API Error processing file {filename}:")
            traceback.print_exc()
            return jsonify({"error": f"An error occurred during prediction: {str(e)}", "status": "error"}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                    print(f"API: Temporary image {temp_image_path} deleted.")
                except Exception as e:
                    print(f"API: Error deleting temporary image {temp_image_path}: {e}")
    else:
        return jsonify({"error": "Invalid file type. Allowed types: png, jpg, jpeg.", "status": "error"}), 400

# --- Main Execution ---
if __name__ == '__main__':
    configure_tensorflow_gpu() 
    if load_application_models(): # Load models when the Flask app starts
        print(f"Flask app starting. Access at http://127.0.0.1:5000")
        # Use host='0.0.0.0' to make the server accessible from other devices on your network.
        # debug=True is useful for development but should be False in production.
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("CRITICAL: Models could not be loaded. The application will not work correctly. Exiting.")
