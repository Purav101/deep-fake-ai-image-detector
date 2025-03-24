from  flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import os

app = Flask(__name__)
CORS(app)

# Configure model path
MODEL_DIR = r'C:\Users\apurv\OneDrive\Desktop\python\ml projects\deepfake\df\deepfake_vs_real_image_detection'

# Load model and processor
model = ViTForImageClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True
)
processor = ViTImageProcessor.from_pretrained(MODEL_DIR)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Open and process image
            image = Image.open(file.stream).convert('RGB')
            
            # Prepare image for model
            inputs = processor(images=image, return_tensors="pt")
            
            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get prediction and confidence
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[0][predicted_class].item() * 100
                
                # Map class to label (adjust based on your model's classes)
                is_deepfake = predicted_class == 1  # Adjust based on your model's class mapping
                
                result = {
                    'is_deepfake': is_deepfake,
                    'confidence': confidence,
                    'message': 'Analysis complete'
                }
                
                return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5500)