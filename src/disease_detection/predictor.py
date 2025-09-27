import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from .ml_model.model import CNNModel

CLASS_NAMES = ['American Bollworm on Cotton', 'Anthracnose on Cotton', 'Army worm', 'bacterial_blight in Cotton', 'Becterial Blight in Rice', 'bollrot on Cotton', 'bollworm on Cotton', 'Brownspot', 'Common_Rust', 'Cotton Aphid', 'cotton mealy bug', 'cotton whitefly', 'Flag Smut', 'Gray_Leaf_Spot', 'Healthy cotton', 'Healthy Maize', 'Healthy Wheat', 'Leaf Curl', 'Leaf smut', 'maize ear rot', 'maize fall armyworm', 'maize stem borer', 'Mosaic sugarcane', 'pink bollworm in cotton', 'red cotton bug', 'RedRot sugarcane', 'RedRust sugarcane', 'Rice Blast', 'Sugarcane Healthy', 'thirps on  cotton', 'Tungro', 'Wheat aphid', 'Wheat black rust', 'Wheat Brown leaf Rust', 'Wheat leaf blight', 'Wheat mite', 'Wheat powdery mildew', 'Wheat scab', 'Wheat Stem fly', 'Wheat___Yellow_Rust', 'Wilt', 'Yellow Rust Sugarcane']

image_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'disease_detection', 'ml_model', 'model.pth')

model = CNNModel(num_classes=len(CLASS_NAMES))

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

model.eval()

def predict_disease(image_file):
    try:
        image = Image.open(image_file).convert("RGB")

        image_tensor = image_transforms(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim = 1)
            top_prob, top_catid = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[top_catid.item()]
        confidence = top_prob.item()

        return predicted_class, confidence
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None