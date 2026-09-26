import torch
from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image

# Load model with correct weights
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# Load labels
with open("app/imagenet_labels.txt") as f:
    LABELS = [line.strip() for line in f.readlines()]

# Correct preprocessing
transform = weights.transforms()

def predict_image(image: Image.Image, top_k: int = 3):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top-k predictions
        confidences, indices = torch.topk(probabilities, top_k)

    results = []
    for conf, idx in zip(confidences, indices):
        results.append({
            "label": LABELS[idx.item()],
            "confidence": round(conf.item() * 100, 2)  # percentage
        })

    return results