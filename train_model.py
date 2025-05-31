import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib # For saving and loading the SVM model and label encoder
import traceback # For more detailed error logging

# --- GPU and TensorFlow Optimization Configuration ---
print("Configuring TensorFlow for GPU and optimizations...")

# Set environment variables for GPU usage (do this before importing TensorFlow if possible, but here is fine too)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0, adjust if you have multiple GPUs and want a specific one
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" # Good practice to allow memory growth

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for gpu_device in gpus: # Renamed gpu to gpu_device to avoid conflict with any potential global 'gpu'
            print(f"  {gpu_device}")
            try:
                tf.config.experimental.set_memory_growth(gpu_device, True)
                print(f"    Memory growth enabled for {gpu_device.name}")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"    Error setting memory growth for {gpu_device.name}: {e}")

        # Enable Mixed Precision (can speed up training and inference on compatible GPUs)
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print(f"Mixed precision policy set to: {policy.name}")
        except Exception as e:
            print(f"Could not set mixed precision policy: {e}")

        # Enable XLA (Accelerated Linear Algebra)
        try:
            tf.config.optimizer.set_jit(True)
            print("XLA (Accelerated Linear Algebra) enabled.")
        except Exception as e:
            print(f"Could not enable XLA: {e}")
    else:
        print("No GPU found by TensorFlow. Using CPU. Ensure CUDA and cuDNN are correctly installed and compatible with your TensorFlow version if you have a GPU.")
except Exception as e:
    print(f"Error during TensorFlow device configuration: {e}")

# CPU Thread Optimization (can be useful if running on CPU or for CPU-bound parts)
try:
    tf.config.threading.set_intra_op_parallelism_threads(4) # Threads for individual operations
    tf.config.threading.set_inter_op_parallelism_threads(4) # Threads for parallel operations
    print("CPU thread optimization configured.")
except Exception as e:
    print(f"Warning: Could not set CPU thread optimization: {e}")


# --- Configuration ---
IMAGE_SIZE = (224, 224) # EfficientNetB0 default input size
CATEGORIES = ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"] # CRUCIAL: Order defines numeric mapping
MODEL_SAVE_PATH_SVM = "apple_leaf_svm_model.joblib"
MODEL_SAVE_PATH_LE = "apple_leaf_label_encoder.joblib"
# FEATURE_EXTRACTOR_SAVE_PATH = "efficientnet_feature_extractor.h5" # Optional
BATCH_SIZE_FEATURES = 32

# --- 1. Load and Preprocess Data ---
def load_images_and_labels(local_categories, image_size_tuple): # Renamed args for clarity
    images_list = []
    labels_list = []
    print("Loading images...")
    for category_idx, category_name in enumerate(local_categories):
        # Use the correct path structure
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), category_name)
        print(f"Checking directory: {path}")
        if not os.path.isdir(path):
            print(f"Warning: Directory not found for category '{category_name}': {path}. Skipping.")
            continue
        class_num = category_idx # This assigns 0, 1, 2, 3 based on CATEGORIES order
        image_files = os.listdir(path)
        if not image_files:
            print(f"Warning: No images found in directory: {path}. Skipping category '{category_name}'.")
            continue

        for img_name in image_files:
            try:
                img_path = os.path.join(path, img_name)
                print(f"Processing image: {img_path}")
                img_array = cv2.imread(img_path)
                if img_array is None:
                    print(f"Warning: Could not read image {img_path}. Skipping.")
                    continue
                img_resized = cv2.resize(img_array, image_size_tuple)
                images_list.append(img_resized)
                labels_list.append(class_num) # labels_list will contain 0, 1, 2, 3
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    if not images_list:
        raise ValueError("No images were loaded. Please check DATASET_PATH and subfolder structure. Aborting.")
    print(f"Successfully loaded {len(images_list)} images.")
    return np.array(images_list), np.array(labels_list)

images, labels_numeric = load_images_and_labels(CATEGORIES, IMAGE_SIZE)

# Encode labels. Since 'labels_numeric' already contains 0,1,2,3 based on CATEGORIES order,
# LabelEncoder().fit_transform() on these numeric labels will result in
# label_encoder.classes_ being [0, 1, 2, 3].
label_encoder = LabelEncoder()
labels_encoded_for_svm = label_encoder.fit_transform(labels_numeric)

try:
    joblib.dump(label_encoder, MODEL_SAVE_PATH_LE)
    print(f"Label encoder saved to {MODEL_SAVE_PATH_LE}")
    # This output confirms the internal state of the saved LabelEncoder
    print(f"  LabelEncoder internal classes_ (should be numeric, e.g., [0 1 2 3]): {label_encoder.classes_}")
    
    # Save the string class names in order for verification by app.py
    # This file explicitly states the mapping from index to string name.
    CLASS_NAMES_FILE = "class_names_from_training.txt"
    with open(CLASS_NAMES_FILE, "w") as f:
        for idx, category_name in enumerate(CATEGORIES):
            f.write(f"{idx}:{category_name}\n") # Format: 0:Apple Scab, 1:Black Rot, ...
    print(f"String class names and their order saved to {CLASS_NAMES_FILE}. This order is: {CATEGORIES}")
except Exception as e:
    print(f"Error saving label encoder or class names file: {e}")
    traceback.print_exc()

# --- 2. Feature Extraction with EfficientNet ---
print("Initializing EfficientNetB0 for feature extraction...")
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), pooling='avg')
feature_extractor_model = Model(inputs=base_model.input, outputs=base_model.output, name="EfficientNetB0_Feature_Extractor")

def extract_features(images_data, model, batch_size):
    print(f"Extracting features with batch size {batch_size}...")
    if not isinstance(images_data, np.ndarray):
        images_data = np.array(images_data)
    images_preprocessed = preprocess_input(images_data.copy())
    features = model.predict(images_preprocessed, batch_size=batch_size, verbose=1)
    print(f"Features extracted. Shape: {features.shape}")
    return features

image_features = extract_features(images, feature_extractor_model, BATCH_SIZE_FEATURES)

# --- 3. Train SVM Classifier ---
# labels_encoded_for_svm contains numeric labels [0, 1, 2, 3]
X_train, X_test, y_train_svm, y_test_svm = train_test_split(
    image_features, labels_encoded_for_svm, test_size=0.2, random_state=42, stratify=labels_encoded_for_svm
)

print(f"Training data shape: {X_train.shape}, Training SVM labels shape: {y_train_svm.shape}")
print(f"Testing data shape: {X_test.shape}, Testing SVM labels shape: {y_test_svm.shape}")

print("Training SVM model...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train, y_train_svm) # SVM is trained on numeric labels 0,1,2,3
print("SVM model training complete.")

try:
    joblib.dump(svm_model, MODEL_SAVE_PATH_SVM)
    print(f"SVM model saved to {MODEL_SAVE_PATH_SVM}")
    # This output confirms the classes the SVM model itself learned.
    print(f"  SVM model internal classes_ (should be numeric, e.g., [0 1 2 3]): {svm_model.classes_}")
    
    SVM_PARAMS_FILE = "svm_params.txt"
    with open(SVM_PARAMS_FILE, "w") as f:
        f.write(f"Kernel: {svm_model.kernel}\nC: {svm_model.C}\nGamma: {svm_model.gamma}\n")
        f.write(f"Probability: {svm_model.probability}\nRandom State: {svm_model.random_state}\n")
        f.write(f"SVM internal classes_: {svm_model.classes_}\n")
    print(f"SVM parameters saved to {SVM_PARAMS_FILE}")
except Exception as e:
    print(f"Error saving SVM model or its parameters: {e}")
    traceback.print_exc()

# --- 4. Evaluate the SVM Model ---
print("Evaluating SVM model...")
y_pred_numeric_svm = svm_model.predict(X_test) # Predictions are numeric indices
accuracy_svm = accuracy_score(y_test_svm, y_pred_numeric_svm)
print(f"SVM Model Accuracy: {accuracy_svm * 100:.2f}%")
# For classification_report, target_names should be the string names corresponding
# to the numeric labels 0,1,2,3 in the order defined by CATEGORIES.
print("\nSVM Classification Report (using string CATEGORIES for names):\n", 
      classification_report(y_test_svm, y_pred_numeric_svm, target_names=CATEGORIES))

# --- 5. Prediction Function (Example for this script, similar to app.py's logic) ---
def local_predict_leaf_disease(image_path, feature_extractor, svm_clf, # Renamed svm_classifier
                               string_categories_list, # Explicitly pass string names
                               img_size_tuple=(224, 224)):
    try:
        img_array = cv2.imread(image_path)
        if img_array is None: return "Error: Could not read image.", None
        img_resized = cv2.resize(img_array, img_size_tuple)
        img_for_nn = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_for_nn.copy())
        
        features_single = feature_extractor.predict(img_preprocessed, verbose=0)
        
        prediction_numeric = svm_clf.predict(features_single)[0] # SVM predicts numeric index
        prediction_proba_all = svm_clf.predict_proba(features_single)
        
        predicted_class_str_name = "Unknown"
        if 0 <= prediction_numeric < len(string_categories_list):
            predicted_class_str_name = string_categories_list[prediction_numeric]
        else:
            print(f"Warning (local_predict): Predicted numeric index {prediction_numeric} out of bounds for string_categories_list.")
            predicted_class_str_name = f"Unknown Class (Index: {prediction_numeric})"
            
        probabilities_map = {}
        if len(prediction_proba_all[0]) == len(string_categories_list):
            for i, class_name_key_str in enumerate(string_categories_list):
                probabilities_map[class_name_key_str] = float(prediction_proba_all[0][i])
        else:
            print(f"Warning (local_predict): SVM proba length mismatch with string_categories_list.")
            # Fallback for probabilities
            for i in range(len(prediction_proba_all[0])):
                key_name = string_categories_list[i] if i < len(string_categories_list) else f"Internal_Class_Index_{i}"
                probabilities_map[key_name] = float(prediction_proba_all[0][i])

        return predicted_class_str_name, probabilities_map
    except Exception as e:
        print(f"Error during local prediction: {e}")
        traceback.print_exc()
        return f"Error: {e}", None

# --- Example of how to load models and predict (for this script's verification) ---
if __name__ == '__main__':
    print("\n--- Example Prediction (from train_model.py) ---")
    try:
        # Models are already in memory if script runs top-to-bottom
        loaded_feature_extractor_local = feature_extractor_model
        # For this example, we'll reload the SVM and LabelEncoder to mimic app.py
        if os.path.exists(MODEL_SAVE_PATH_SVM) and os.path.exists(MODEL_SAVE_PATH_LE):
            loaded_svm_model_local = joblib.load(MODEL_SAVE_PATH_SVM)
            # loaded_label_encoder_local = joblib.load(MODEL_SAVE_PATH_LE) # Not strictly needed for string mapping if using CATEGORIES
            print("Models reloaded for example prediction.")
            print(f"  Reloaded SVM model internal classes_: {loaded_svm_model_local.classes_}")
            # print(f"  Reloaded LabelEncoder internal classes_: {loaded_label_encoder_local.classes_}")

            if images.size > 0:
                first_category_name_for_test = CATEGORIES[0] # e.g., "Apple Scab"
                first_category_path_for_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), first_category_name_for_test)
                if os.path.exists(first_category_path_for_test) and os.listdir(first_category_path_for_test):
                    example_image_name_for_test = os.listdir(first_category_path_for_test)[0]
                    example_image_path_for_test = os.path.join(first_category_path_for_test, example_image_name_for_test)
                    
                    print(f"Predicting for image: {example_image_path_for_test}")

                    # Use the local_predict_leaf_disease function with the global CATEGORIES list
                    predicted_class_string, probabilities_string_keys = local_predict_leaf_disease(
                        example_image_path_for_test,
                        loaded_feature_extractor_local,
                        loaded_svm_model_local,
                        CATEGORIES, # Pass the string category names
                        IMAGE_SIZE
                    )

                    if probabilities_string_keys:
                        print(f"Predicted Class (String): {predicted_class_string}")
                        print("Probabilities (String Keys):")
                        for cls_str_key, prob_val_num in probabilities_string_keys.items():
                            print(f"  {cls_str_key}: {prob_val_num*100:.2f}%")
                    else:
                        print(f"Prediction failed. Message: {predicted_class_string}")
                else:
                    print(f"Could not find an example image in {first_category_path_for_test}")
            else:
                print("No images were loaded, cannot run example prediction.")
        else:
            print("Error: Model files (.joblib) not found. Cannot run example prediction. Please ensure training completed successfully.")

    except FileNotFoundError as fnf_error:
        print(f"Error in example prediction: Model file not found. {fnf_error}")
    except Exception as e:
        print(f"An error occurred during example prediction: {e}")
        traceback.print_exc()