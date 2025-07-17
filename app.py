from flask import Flask, render_template, request, redirect, url_for
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os

# --- Flask App Initialization ---
app = Flask(__name__, 
            template_folder='templates', 
            static_folder='static')

# --- Configuration (Adjust these based on your model and data) ---
MODEL_PATH = r'D:\models\braille_recognition_model.keras' 
TARGET_IMAGE_SIZE = (240, 240)
LABELS = [chr(ord('a') + i) for i in range(26)] 

# --- Model Loading (Initialize outside of route handlers) ---
model = None

def load_model_once():
    """Loads the Keras model. Handles potential errors during loading."""
    global model
    if model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
            
            model = keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            model = None
    return model

# --- Image Preprocessing Function ---
def preprocess_image(image_bytes):
    """
    Preprocesses the image for the model.
    Args:
        image_bytes: The byte data of the uploaded image.
    Returns:
        A preprocessed NumPy array (shape: (1, H, W, C)), or None on error.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(TARGET_IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the landing page."""
    return render_template('index.html')

@app.route('/translator', methods=['GET'])
def translator_page():
    """Renders the Braille translator page with the upload form."""
    return render_template('Translator.html', prediction=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles the image upload, preprocessing, prediction, and result display.
    Results are now displayed on Translator.html.
    """
    current_model = load_model_once()

    if current_model is None:
        return render_template('error.html', message='Model failed to load. Please check server logs.')

    if 'file' not in request.files:
        return render_template('error.html', message='No file part in the request.')
    
    file = request.files['file']

    if file.filename == '':
        return render_template('error.html', message='No file selected.')

    if file:
        try:
            image_bytes = file.read()
            processed_image = preprocess_image(image_bytes)
            
            if processed_image is None:
                return render_template('error.html', message='Error processing the image. Please upload a valid image file.')

            prediction = current_model.predict(processed_image)
            predicted_class_index = np.argmax(prediction)
            
            if 0 <= predicted_class_index < len(LABELS):
                predicted_class_label = LABELS[predicted_class_index]
            else:
                predicted_class_label = "Unknown (Index out of bounds)"
                print(f"Warning: Predicted class index {predicted_class_index} is out of bounds for LABELS.")

            return render_template('Translator.html', prediction=predicted_class_label)

        except Exception as e:
            print(f"Exception during upload/prediction: {e}")
            return render_template('error.html', message=f'An unexpected error occurred: {e}')
    
    return render_template('error.html', message='An unknown error occurred.')

@app.route('/error')
def show_error():
    message = request.args.get('message', 'An unknown error occurred.')
    return render_template('error.html', message=message)

# --- Placeholder Routes for other HTML files ---
@app.route('/about', methods=['GET'])
def about_page():
    """Renders the about page."""
    return render_template('about.html') 

@app.route('/contactus', methods=['GET'])
def contactus_page():
    """Renders the contact us page."""
    return render_template('ContactUs.html') 

if __name__ == '__main__':
    load_model_once()  
    
    if model is not None:
        # --- DEBUGGING STEP: Print all registered endpoints ---
        print("\n--- Flask Registered Endpoints ---")
        for rule in app.url_map.iter_rules():
            print(f"Endpoint: {rule.endpoint}, Methods: {rule.methods}, Rule: {rule.rule}")
        print("----------------------------------\n")
        # --- END DEBUGGING STEP ---

        app.run(debug=True, host='0.0.0.0')
    else:
        print("Failed to start the application because the model could not be loaded.")
