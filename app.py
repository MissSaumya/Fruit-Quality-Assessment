import os
import random
import shutil
from flask import Flask, render_template, request, redirect
from model_helper import load_model, predict

app = Flask(__name__)

# --- CONFIGURATION ---
# Using raw strings (r"") to prevent Windows path errors
MODEL_PATH = r"C:\Users\mishr\00000\FruitVision Augmented\sgd momentum\vision_sgd_pretrainedFalse_cosine_model_combined.pth"
DATASET_PATH = r"D:\Research\Datasets\FruitVision\Augmented-Resized Image"

# Flask requires images to be in the 'static' folder to display them in HTML
STATIC_IMG_PATH = os.path.join('static', 'current_img.jpg')

# Load the SOTA ViTTwoHead model into memory
model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return redirect('/')
    file = request.files['file']
    if file.filename == '': return redirect('/')
    
    # Save uploaded file for display
    file.save(STATIC_IMG_PATH)
    
    # Predict using the multi-head architecture
    fruit_p, qual_p, fruit_c, qual_c = predict(model, STATIC_IMG_PATH)
    
    # Combine results for the UI
    prediction_text = f"{fruit_p} ({qual_p})"
    avg_confidence = (fruit_c + qual_c) / 2
    
    return render_template('index.html', 
                           prediction=prediction_text, 
                           confidence=f"{avg_confidence:.2f}%", 
                           mode="Manual Upload")

@app.route('/random_sample')
def random_sample():
    # 1. Level 1: Navigate to a random Fruit Type folder
    fruits = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    selected_fruit = random.choice(fruits)
    
    # 2. Level 2: Navigate to a random Quality folder (Fresh, Rotten, or Formalin-mixed)
    fruit_dir = os.path.join(DATASET_PATH, selected_fruit)
    qualities = [d for d in os.listdir(fruit_dir) if os.path.isdir(os.path.join(fruit_dir, d))]
    selected_quality = random.choice(qualities)
    
    # 3. Level 3: Select a random image from the final subfolder
    final_dir = os.path.join(fruit_dir, selected_quality)
    images = [f for f in os.listdir(final_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_img = random.choice(images)
    
    img_full_path = os.path.join(final_dir, selected_img)
    
    # Copy selected image to static folder for web display
    shutil.copy(img_full_path, STATIC_IMG_PATH)
    
    # 4. Model Prediction (Returns 4 values from the two heads)
    fruit_pred, qual_pred, f_conf, q_conf = predict(model, img_full_path)
    
    # 5. UI Formatting
    prediction_text = f"{fruit_pred} ({qual_pred})"
    avg_conf = (f_conf + q_conf) / 2
    
    # 6. Strict Validation Logic
    # We compare the folder names against the model's textual output
    # Note: 'formalin-mixed' folder must match 'Formalin-mixed' prediction
    is_correct_fruit = selected_fruit.lower() == fruit_pred.lower()
    is_correct_qual = selected_quality.lower() == qual_pred.lower()
    
    if is_correct_fruit and is_correct_qual:
        is_correct = "Correct"
    else:
        actual_label = f"{selected_fruit} ({selected_quality})"
        is_correct = f"Incorrect (Actual: {actual_label})"

    return render_template('index.html', 
                           prediction=prediction_text, 
                           confidence=f"{avg_conf:.2f}%", 
                           is_correct=is_correct, 
                           mode="Random Test")

if __name__ == '__main__':
    if not os.path.exists('static'): os.makedirs('static')
    app.run(debug=True)