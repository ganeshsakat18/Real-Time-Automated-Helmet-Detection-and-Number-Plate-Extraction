import os
import cv2
from yolov5 import YOLOv5
import easyocr
from excel_logger import log_to_excel
from email_notification import send_challan_email  # Ensure the correct import path
from ultralytics import YOLO
from PIL import Image


# Load YOLOv5 models
helmet_model = YOLO(r"C:\Users\user\Desktop\folder\detection_models\yolov8n.pt")  # Helmet detection model
number_plate_model = YOLOv5(r"C:\Users\user\Desktop\folder\detection_models\rider_helmet_number_medium.pt")  # Number plate and head model

# OCR Reader for number plate text extraction
reader = easyocr.Reader(['en'])

# Output folder paths
RECIPIENT_EMAIL = "vrx2407@gmail.com"  # Replace with the recipient's email

output_image_folder = r'C:\Users\user\Desktop\folder\output\images'
output_video_folder = r'C:\Users\user\Desktop\folder\output\videos'
output_realtime_folder = r'C:\Users\user\Desktop\folder\output\real_time'

# Create output folders if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_video_folder, exist_ok=True)
os.makedirs(output_realtime_folder, exist_ok=True)

# Specific folders for head and number plate detection
head_folder = r'C:\Users\user\Desktop\folder\output\owner_photos'
number_plate_folder = r'C:\Users\user\Desktop\folder\output\number_plates'
os.makedirs(head_folder, exist_ok=True)
os.makedirs(number_plate_folder, exist_ok=True)


def detect_helmet_and_plate(frame, image_name="default_image"):
    """
    Detect helmet, number plate, and driver head in the provided frame.
    Creates separate folders for the image name under head and number plate folders.
    """
    helmet_detected = False
    plates = []

    # Create unique folders for the current image
    image_folder_name = os.path.splitext(image_name)[0]
    current_head_folder = os.path.join(head_folder, image_folder_name)
    current_plate_folder = os.path.join(number_plate_folder, image_folder_name)
    os.makedirs(current_head_folder, exist_ok=True)
    os.makedirs(current_plate_folder, exist_ok=True)

    # Helmet Detection
    results_helmet = helmet_model(frame)  # Run YOLO model on the frame

    if hasattr(results_helmet, 'xywh'):
        # Process results in xywh format (center x, center y, width, height)
        for result in results_helmet.xywh[0]:
            x_center, y_center, width, height, conf, cls = result
            if conf > 0.5 and int(cls) == 0:  # Assuming class 0 is for helmets
                x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
                x2, y2 = int(x_center + width / 2), int(y_center + height / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                helmet_detected = True

    # Number Plate and Driver Head Detection
    results_number_plate = number_plate_model.predict(frame)

    if hasattr(results_number_plate, 'xyxy'):
        for result in results_number_plate.xyxy[0]:  # Use xyxy format for YOLOv5
            x1, y1, x2, y2, conf, cls = result
            if conf > 0.5:
                if int(cls) == 2:  # Number plate class
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]
                    ocr_results = reader.readtext(cropped_plate)
                    if ocr_results:
                        for _, text, prob in ocr_results:
                            if prob > 0.5:
                                plates.append(text)

                    # Save cropped number plate image
                    plate_img_path = os.path.join(current_plate_folder, f"plate_{int(x1)}_{int(y1)}.jpg")
                    cv2.imwrite(plate_img_path, cropped_plate)

                elif int(cls) == 1:  # Driver head class
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    head_img = frame[int(y1):int(y2), int(x1):int(x2)]
                    head_img_path = os.path.join(current_head_folder, f"head_{int(x1)}_{int(y1)}.jpg")
                    cv2.imwrite(head_img_path, head_img)

    plates.clear()

    # Function to process images in the folder
    def process_images_in_folder(folder_path):
        # List all image files in the folder
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(tuple(image_extensions))]

        # Loop through all image files
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)

            print(f"Processing image: {image_file}")

            # Step 4: Load the cropped number plate image
            img = Image.open(image_path)

            # Step 5: Apply OCR to extract the number plate text using EasyOCR
            ocr_result = reader.readtext(img)
            # Extract text from OCR result and append it to the plates list
            for _, text, prob in ocr_result:
                if prob > 0.5:
                    plates.append(text)

    # Example usage:
    # Replace this with the folder path containing your pre-cropped number plate images
    folder_path = current_plate_folder

    # Process all images in the folder
    process_images_in_folder(folder_path)
   



    print("Helmet Detected:", helmet_detected)
    print("Detected Plates:", plates)

    return frame, helmet_detected, plates

#dashboard
   

def process_image(image_path):
    """
    Process a single image for helmet and plate detection.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        image_name = os.path.basename(image_path)
        processed_image, helmet_detected, plates = detect_helmet_and_plate(image, image_name)

        # Save the processed image
        output_image_path = os.path.join(output_image_folder, os.path.basename(image_path))
        cv2.imwrite(output_image_path, processed_image)
        print(f"Processed image saved at: {output_image_path}")

        # Log the detection result
        log_to_excel(image_path, helmet_detected, plates, "Detected")

        # Send challan if a helmet violation is detected
        if not helmet_detected:
            print("Helmet violation detected!")
            send_challan_email(output_image_path, plates, RECIPIENT_EMAIL)

    except Exception as e:
        print(f"Error in processing image: {e}")


def process_video(video_path):
    """
    Process a video for helmet and plate detection.
    """
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise FileNotFoundError(f"Video not found at path: {video_path}")

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = os.path.join(output_video_folder, os.path.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            processed_frame, helmet_detected, plates = detect_helmet_and_plate(frame)
            out_video.write(processed_frame)

            

            # Log detection result
            log_to_excel(video_path, helmet_detected, plates, "Detected")

            # Send challan if helmet violation is detected
            if not helmet_detected:
                print("Helmet violation detected in video frame!")
                send_challan_email(output_video_path, plates, RECIPIENT_EMAIL)

        video_capture.release()
        out_video.release()
        print(f"Processed video saved at: {output_video_path}")

    except Exception as e:
        print(f"Error in processing video: {e}")


def start_real_time_feed():
    """
    Start real-time camera feed and process frames for helmet and plate detection.
    """
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Unable to access the camera.")

        fps = 20.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_realtime_path = os.path.join(output_realtime_folder, 'realtime_output.avi')

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_realtime = cv2.VideoWriter(output_realtime_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, helmet_detected, plates = detect_helmet_and_plate(frame)
            out_realtime.write(processed_frame)

            # Log detection result
            log_to_excel("Real-Time", helmet_detected, plates, "Detected")

            # Send challan if a helmet violation is detected
            if not helmet_detected:
                print("Helmet violation detected in real-time feed!")
                send_challan_email("Real-Time", plates, RECIPIENT_EMAIL)

            cv2.imshow("Real-Time Feed", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out_realtime.release()
        cv2.destroyAllWindows()
        print(f"Real-time processed video saved at: {output_realtime_path}")

    except Exception as e:
        print(f"Error in real-time feed: {e}")
