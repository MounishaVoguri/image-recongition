# Image Recognition System

A Python-based image recognition application that uses a **pre-trained ResNet50 deep learning model** to classify images. The application allows users to select an image from their computer and predicts the most likely object present in the image along with its confidence score.

## Features

* Select an image using a simple Tkinter file dialog.
* Preprocess images automatically before prediction.
* Uses the pre-trained **ResNet50** model from Torchvision.
* Classifies images using ImageNet classes.
* Displays the predicted class and confidence score.
* Uses GPU-independent PyTorch inference with gradient calculation disabled.

## How It Works

The application follows these steps:

1. The user selects an image from their computer.
2. The image is resized and center-cropped to the required input size.
3. The image is converted into a PyTorch tensor.
4. ImageNet normalization is applied.
5. The pre-trained ResNet50 model processes the image.
6. Softmax is used to calculate class probabilities.
7. The class with the highest probability is selected as the prediction.
8. The predicted class and confidence score are displayed.

## Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **ResNet50**
* **PIL (Pillow)**
* **Tkinter**
* **ImageNet**

## Example Output

```text
Predicted class: golden retriever
Confidence: 0.9234
```

The actual prediction and confidence score depend on the image provided.

## Project Structure

```text
Image-Recognition-System/
│
├── image_recognition.py
├── imagenet_class_index.json
└── README.md
```

## Installation

Install the required Python libraries:

```bash
pip install torch torchvision pillow
```

Tkinter is generally included with standard Python installations on Windows.

## Running the Project

1. Clone or download the repository.
2. Place `imagenet_class_index.json` in the required project location.
3. Open the Python file.
4. Run the program:

```bash
python image_recognition.py
```

5. Select an image when the file dialog appears.
6. The predicted object and confidence score will be displayed in the terminal.

## Model

This project uses **ResNet50**, a convolutional neural network available through Torchvision with pre-trained ImageNet weights.

The model is used for inference only; no additional model training is performed in this project.

## Future Improvements

* Add a graphical interface to display the selected image and prediction.
* Display the top 5 predictions instead of only the top prediction.
* Add confidence visualization.
* Support webcam-based real-time recognition.
* Improve error handling for unsupported or corrupted images.
* Deploy the application as a standalone desktop application.

## Purpose

This project demonstrates the practical use of **deep learning, computer vision, image preprocessing, and pre-trained models** to build a simple image classification application.

## Author

**Voguri Mounisha**

Computer Science Student | AI & Machine Learning Enthusiast
