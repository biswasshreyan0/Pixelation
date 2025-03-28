import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from picamera2 import Picamera2, Preview
from tensorflow.keras.models import load_model

import base64
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

# Set preferred image size for prediction
PREF_SIZE = (128, 128)

# -------------------------------
# Load YOLOv5 model for object detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.classes = [0]

# -------------------------------
def detect_objects(image_path):
    """
    Perform object detection on the given image using YOLOv5.
    
    Returns:
        list: List of bounding boxes in normalized format [x_center, y_center, norm_width, norm_height].
    """
    img = Image.open(image_path)
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Run detection with YOLOv5
    results = yolo_model(img, size=640)
    results.show()
    
    boxes = print_and_save_bounding_boxes(results, image_path)
    
    num_bounding_boxes = len(results.xyxy[0])
    print(f"Number of Objects Detected: {num_bounding_boxes}")
    
    # Draw bounding boxes on image (for visualization)
    for i in range(num_bounding_boxes):
        x, y, x_plus_w, y_plus_h, _, _ = results.xyxy[0][i]
        draw_bounding_box(
            img_array,
            i + 1,
            round(x.item()),
            round(y.item()),
            round(x_plus_w.item()),
            round(y_plus_h.item())
        )
    return boxes

# -------------------------------
def print_and_save_bounding_boxes(results, image_path):
    """
    Extract bounding boxes from detection results and save them to a text file.
    
    Returns:
        list: List of bounding boxes [x_center, y_center, norm_width, norm_height].
    """
    img = Image.open(image_path)
    image_width, image_height = img.size
    img_filename = os.path.splitext(os.path.basename(image_path))[0]
    txt_filename = f"{img_filename}.txt"
    
    boxes = []
    with open(txt_filename, "w") as f:
        for i, detection in enumerate(results.pandas().xyxy[0].values):
            x_min, y_min, x_max, y_max, confidence, class_id, *_ = detection.tolist()
            if confidence > 0.4:
                label = results.names[int(class_id)]
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                norm_width = (x_max - x_min) / image_width
                norm_height = (y_max - y_min) / image_height
                print(f"Labels: {x_center:.2f} {y_center:.2f} {norm_width:.2f} {norm_height:.2f} {confidence:.2f}")
                print(f"Object {i + 1}: {label} (Confidence: {confidence:.2f})")
                f.write(f"{label} {x_center} {y_center} {norm_width} {norm_height} {confidence}\n")
                boxes.append([x_center, y_center, norm_width, norm_height])
    return boxes

# -------------------------------
def draw_bounding_box(img_array, object_number, x, y, x_plus_w, y_plus_h):
    """
    Draw a bounding box on the image array.
    """
    color = (0, 255, 0)  # Green
    cv2.rectangle(img_array, (x, y), (x_plus_w, y_plus_h), color, 2)

# -------------------------------
def predictions(m_model, image_path):
    """
    Process the image, perform object detection, and run fall detection prediction using the provided model.
    
    Returns:
        bool: True if a fall is detected, otherwise False.
    """
    # Read image for cropping (using matplotlib for consistency)
    img = plt.imread(image_path)
    bounding_boxes = detect_objects(image_path)
    print("Bounding Boxes:", bounding_boxes)
    complete_images = []
    is_fall_detected = False

    # Crop detected regions based on bounding boxes
    for box in bounding_boxes:
        image_height, image_width, _ = img.shape
        xmin, ymin, width, height = box
        xmin = int(xmin * image_width)
        ymin = int(ymin * image_height)
        width = int(width * image_width)
        height = int(height * image_height)
        
        # Ensure indices remain within bounds
        cropped = img[
            max(ymin - height // 2, 0):min(ymin + height // 2, image_height),
            max(xmin - width // 2, 0):min(xmin + width // 2, image_width)
        ]
        complete_images.append(cropped)
    
    # Process each cropped region and predict fall
    for cropped_img in complete_images:
        plt.imshow(cropped_img)
        plt.axis('off')
        plt.show()
        
        cropped_img_resized = cv2.resize(cropped_img, PREF_SIZE)
        plt.imshow(cropped_img_resized)
        plt.axis('off')
        plt.show()
        
        cropped_img_resized = cropped_img_resized / 255.0
        cropped_img_resized = np.expand_dims(cropped_img_resized, axis=0)
        pred = m_model.predict(cropped_img_resized)
        k = np.argmax(pred)
        print("Prediction:", k)
        if k == 0:
            is_fall_detected = True
            print("Fall detected")
            break
        elif k == 1:
            print("No fall detected. Person is walking or standing")
        else:
            print("No fall detected. Person is sitting.")
    return is_fall_detected


# -------------------------------
def send_email_with_sendgrid(image_path):
    # Define the email details
    message = Mail(
        from_email='sujoy.biswas@gmail.com',
        to_emails='biswasshreyan0@gmail.com',
        subject='Fall Alert',
        html_content='<strong>A fall has been detected. Please see the attached image.</strong>'
    )
    
    # Open the image file in binary mode and encode it in base64
    with open(image_path, 'rb') as f:
        data = f.read()
    encoded_file = base64.b64encode(data).decode()

    # Create the attachment and set its properties
    attachedFile = Attachment()
    attachedFile.file_content = FileContent(encoded_file)
    attachedFile.file_type = FileType('image/jpeg')  # adjust if using a different format (e.g., 'image/png')
    attachedFile.file_name = FileName('fall_detected.jpg')
    attachedFile.disposition = Disposition('attachment')
    
    # Attach the file to the message
    message.attachment = attachedFile

    try:
        # Initialize the SendGrid client with your API key
        #sg = SendGridAPIClient('YOUR_SENDGRID_API_KEY')
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print("Email sent with status code:", response.status_code)
        time.sleep(10000)
    except Exception as e:
        print("Error sending email:", e)

# -------------------------------
def main():
    """
    Main function:
      1. Loads the fall detection model.
      2. Captures images from the camera every 2 seconds.
      3. Calls the predictions function to detect a fall.
      4. If a fall is detected, monitors for 1 minute and sends an email alert if the fall persists.
    """
    # Load the fall detection model once
    input_shape = (128, 128, 3)
    model_path = '/home/shreyan/Downloads/MobileNetV2v2.keras'
    m_model = load_model(model_path, compile=False, custom_objects={'input_shape': input_shape})
    
    # Initialize and configure the camera
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (800, 600)})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(2)  # Allow time for camera warm-up
    
    try:
        while True:
            image_path = "current.jpg"
            metadata = picam2.capture_file(image_path)
            print("Captured image:", metadata)
            
            # Call the modularized prediction function from the previous module
            if predictions(m_model, image_path):
                print("Fall detected. Monitoring for persistent fall...")
                # Save the image when fall is first detected
                stored_image = "fall_detected.jpg"
                shutil.copy(image_path, stored_image)
                
                # Monitor for 1 minute (checking every 2 seconds)
                persistent = True
                start_time = time.time()
                while time.time() - start_time < 4:
                    time.sleep(4)
                    metadata = picam2.capture_file(image_path)
                    if not predictions(m_model, image_path):
                        persistent = False
                        print("Fall no longer detected during monitoring.")
                        break
                if persistent:
                    print("Fall persisted for 1 minute. Sending alert email...")
                    #send_email_alert(stored_image)
                    send_email_with_sendgrid('fall_detected.jpg')
                    print("Email sent. going to sleep for 10 sec!")
                    time.sleep(10)
                else:
                    print("Fall was not persistent. No email sent.")
            else:
                print("No fall detected.")
                
            # Wait before the next capture
            time.sleep(4)
            
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        picam2.close()

if __name__ == "__main__":
    main()
