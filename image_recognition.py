import torch
from torchvision import models, transforms
from PIL import Image
import json
import tkinter as tk
from tkinter import filedialog

# Load class labels (e.g., ImageNet)
def load_labels():
    with open('C:\\Users\\LENOVO\\OneDrive\\Documents\\imagenet_class_index.json') as f:
        labels = json.load(f)
    return [labels[str(i)][1] for i in range(len(labels))]

# Load and preprocess the input image
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path)
    img = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img

# Load a pre-trained model (only once)
def load_model():
    from torchvision.models import ResNet50_Weights
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()  # Set model to evaluation mode
    return model

# Make predictions
def predict_image(model, input_tensor):
    with torch.no_grad():  # Disable gradient calculation
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

# Function to make the prediction accessible
def recognize_image(image_path, model, labels):
    input_tensor = preprocess_image(image_path)
    probabilities = predict_image(model, input_tensor)
    top_prob, top_catid = torch.topk(probabilities, 1)

    predicted_class = labels[top_catid.item()]
    confidence = top_prob.item()

    return predicted_class, confidence

# Load the model and labels once
model = load_model()
labels = load_labels()

# Open a Tkinter file dialog to select the image
root = tk.Tk()
root.withdraw()  # Hide the root window
image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])

# Make sure the user selected a file
if image_path:
    # Make the prediction
    predicted_class, confidence = recognize_image(image_path, model, labels)

    # Print the result
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
else:
    print("No file selected!")
