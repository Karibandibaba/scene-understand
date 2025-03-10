import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load YOLOv8 model for object detection
model = YOLO("yolov8n.pt")

# Load BLIP model for scene understanding
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to detect objects
def detect_objects(image):
    results = model(image)
    objects = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            label = model.names[int(box.cls[0])]  # Object class label
            conf = float(box.conf[0])  # Confidence score

            objects.append({"label": label, "bbox": (x1, y1, x2, y2), "confidence": conf})

    return objects

# Function to draw bounding boxes
def draw_objects(image, objects):
    img = np.array(image)
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = obj["label"]
        confidence = obj["confidence"]

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({confidence:.2f})"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

# Function for scene understanding (image captioning)
def generate_caption(image):
    image = image.convert("RGB")  # Ensure correct format
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        caption_ids = blip_model.generate(**inputs)
    caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption

# Streamlit UI
st.title("Scene Understanding with Object Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Scene understanding - generate a caption
    caption = generate_caption(image)
    st.subheader("Scene Understanding:")
    st.write(f"**{caption}**")

    # Detect objects
    objects = detect_objects(image)

    # Draw bounding boxes
    result_image = draw_objects(image, objects)
    st.image(result_image, caption="Detected Objects", use_container_width=True)

    # Show object names
    st.subheader("Detected Objects:")
    for obj in objects:
        st.write(f"- **{obj['label']}** (Confidence: {obj['confidence']:.2f})")
