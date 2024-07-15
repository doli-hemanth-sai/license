import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function for license plate detection using YOLOv8 and OCR
def detect_license_plate(image, model):
    results = model(image)
    detected_plates = []
    plate_texts = []
    coordinates = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract the bounding box coordinates
            plate_image = image[y1:y2, x1:x2]  # Crop the license plate from the image
            detected_plates.append(plate_image)
            coordinates.append((x1, y1, x2, y2))

            # Perform OCR on the license plate image
            plate_text = reader.readtext(plate_image, detail=0)
            plate_texts.append(plate_text[0] if plate_text else "")

    return detected_plates, plate_texts, coordinates

def process_video(video_file, model):
    cap = cv2.VideoCapture(video_file.name)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output.mp4v', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform license plate detection and OCR
        detected_plates, plate_texts, coordinates = detect_license_plate(rgb_frame, model)

        # Draw bounding boxes and OCR text on the frame
        for (x1, y1, x2, y2), plate_text in zip(coordinates, plate_texts):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Write the frame into the output video file
        out.write(frame)

        # Display processed frame
        st.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    st.title("License Plate Detection and OCR")

    # Select input type: Image or Video
    input_type = st.selectbox("Select input type", ["Image", "Video"])

    if input_type == "Image":
        # File uploader for image
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                # Read the uploaded image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Display the uploaded image
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Load the YOLOv8 model
                model_path = 'best100.pt'
                model = YOLO(model_path)

                # Perform license plate detection and OCR
                detected_plates, plate_texts, coordinates = detect_license_plate(image, model)

                # Display results
                for i, plate_image in enumerate(detected_plates):
                    st.image(plate_image, caption=f'Detected Plate {i+1}', use_column_width=True)

                    # Display OCR result
                    st.write(f"License Plate {i+1} Text: {plate_texts[i]}")

            except Exception as e:
                st.write("Error: ", e)

    elif input_type == "Video":
        # File uploader for video
        video_file = st.file_uploader("Upload a video", type=["mp4"])
        if video_file is not None:
            try:
                # Load the YOLOv8 model
                model_path = 'best100.pt'
                model = YOLO(model_path)

                # Process the video
                process_video(video_file, model)

            except Exception as e:
                st.write("Error: ", e)

if __name__ == "__main__":
    main()
