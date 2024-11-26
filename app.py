from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import os

device = torch.device("cpu")
print(f"Device: {device}")

model_path  = "model_epoch_48_final.pth"

def load_test_model(model_path):
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 15)
    model = model.to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully from", model_path)
    else:
        print(f"Error: Model file {model_path} not found.")
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    model.eval()
    return model

model = load_test_model(model_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img = Image.open(file.stream).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted_class = torch.max(outputs, 1)

    disease_names = ['Alphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common root Rot',  
                     'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew', 'Mite',
                     'Septoria', 'Smut', 'Stemfly', 'Tanspot', 'Yellow Rust']
    
    predicted_disease = disease_names[predicted_class.item()]
    
    return jsonify({'predicted_disease': predicted_disease})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=5000)
