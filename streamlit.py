import streamlit as st
import tempfile
import os
import cv2
from PIL import Image
import numpy as np
import base64
import YOLO from Ultralytics

# Dummy object detection function (replace with your model)
def detect_objects_on_image(image):
    # Dummy rectangle for demonstration
    output = image.copy()
    height, width = output.shape[:2]
    cv2.rectangle(output, (width//4, height//4), (width*3//4, height*3//4), (0, 255, 0), 3)
    return output

def detect_objects_on_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detected_frame = detect_objects_on_image(frame)
        out.write(detected_frame)
    cap.release()
    out.release()

def get_download_link(file_path, label="Download"):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{label}</a>'
    return href

# Streamlit UI
st.title("📦 Object Detection App")

file = st.file_uploader("Upload an Image or Video", type=['jpg', 'jpeg', 'png', 'mp4'])

if file:
    file_type = file.type
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(file.read())
    temp_file.close()

    if file_type.startswith("image"):
        image = Image.open(temp_file.name).convert("RGB")
        image_np = np.array(image)
        detected_image = detect_objects_on_image(image_np)

        st.image(detected_image, caption="Detected Image", channels="BGR")

        # Save result for download
        result_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        cv2.imwrite(result_path, detected_image)
        st.markdown(get_download_link(result_path, "Download Detected Image"), unsafe_allow_html=True)

    elif file_type == "video/mp4":
        st.video(temp_file.name)

        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        with st.spinner("Processing video..."):
            detect_objects_on_video(temp_file.name, output_video_path)
        st.video(output_video_path)
        st.markdown(get_download_link(output_video_path, "Download Detected Video"), unsafe_allow_html=True)
