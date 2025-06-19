import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

model = models.resnet18(pretrained=True)
model.eval()

with open("app/imagenet_labels.txt") as f:
    LABELS = [line.strip() for line in f.readlines()]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def predict_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        return LABELS[predicted.item()], confidence.item()
