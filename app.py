import streamlit as st
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from torchvision.io import read_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import io

model = load_model(r"C:\MAJOR\ResNet.h5")  # Ensure this is your fine-tuned ResNet model
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size).convert("RGB")  # Resize and convert to RGB
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_tb(img, model, threshold=0.5):
    processed_image = preprocess_image(img, target_size=model.input_shape[1:3])  
    prediction = model.predict(processed_image)  
    if prediction.shape[-1] == 2:  
        predicted_class = np.argmax(prediction)  
        probability = prediction[0][predicted_class]
    else:  
        probability = prediction[0][0]
        predicted_class = int(probability > threshold)
    class_label = "Tuberculosis" if predicted_class == 1 else "Normal"
    return probability, class_label

def generate_smoothgradcam(img_pil):
    torch_model = resnet18(pretrained=True).eval()
    img = transforms.ToTensor()(img_pil)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    cam_extractor = SmoothGradCAMpp(torch_model, target_layer="layer4")
    out = torch_model(input_tensor)
    class_idx = out.squeeze(0).argmax().item()
    activation_map = cam_extractor(class_idx, out)
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    return result

st.title("Tuberculosis Detection and visualization")

name = st.text_input("Name")
age = st.number_input("Age", min_value=0, max_value=120, step=1)
sex = st.selectbox("Sex", ["Male", "Female", "Other"])
medical_history = st.text_area("Medical History")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    probability, label = predict_tb(img, model)
    st.write(f"**Prediction:** {label} ({probability:.2f} confidence)")
    if label == "Tuberculosis":
        st.write("### Tuberculosis detected! Generating SmoothGradCAM++ visualization...")
        cam_img = generate_smoothgradcam(img)
        st.image(cam_img, caption="SmoothGradCAM++ Visualization", use_column_width=True)
    else:
        st.write("### Normal case detected.")