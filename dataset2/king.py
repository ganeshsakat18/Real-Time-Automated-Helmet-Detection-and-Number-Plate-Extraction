import torch
from pathlib import Path
import cv2
from PIL import Image
import easyocr  # Import EasyOCR

# Step 1: Load the YOLOv5 model
# Replace 'path_to_your_model.pt' with the actual path to your trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\user\Desktop\folder\dataset2\rider_helmet_number_medium.pt')  # Load the trained YOLOv5 model

# Check the model structure
print(model)

# Step 2: Load and process the image
image_path = r'C:\Users\user\Desktop\folder\dataset2\input\king.jpg'  # Replace with your image path
img = Image.open(image_path)

# Perform inference on the image
results = model(img)

# Step 3: Extracting the detection results (detected objects, their classes, and bounding boxes)
# results.xywh[0] contains detections (xywh coordinates, confidence, class, etc.)
# Filter results for 'number' class (class 2 corresponds to number plate)
number_plate_results = results.xywh[0][results.xywh[0][:, 5] == 2]  # 5th column is class label, 2 is 'number' class

# Display the filtered results (bounding box coordinates for number plates)
print("Number Plate Detections (Bounding Boxes):")
print(number_plate_results)

# Step 4: Load the image for OpenCV (to crop number plate regions)
img_cv = cv2.imread(image_path)

# Initialize EasyOCR reader (default is English)
reader = easyocr.Reader(['en'])

# Step 5: Loop through the detected number plates and crop them
for index, row in enumerate(number_plate_results):
    x1, y1, w, h = row[0], row[1], row[2], row[3]  # Get the x, y, width, height of the bounding box

    # Convert from xywh to xmin, ymin, xmax, ymax for OpenCV cropping
    xmin, ymin, xmax, ymax = int(x1 - w / 2), int(y1 - h / 2), int(x1 + w / 2), int(y1 + h / 2)

    # Crop the image to get the number plate region
    cropped_img = img_cv[ymin:ymax, xmin:xmax]

    # Optionally, save the cropped number plate image
    cv2.imwrite(f'number_plate_{index}.jpg', cropped_img)

    # Step 6: Display the cropped image
    cv2.imshow(f'Number Plate {index}', cropped_img)

    # Step 7: Apply OCR to extract the number plate text using EasyOCR
    # EasyOCR can directly process the image and extract text
    result = reader.readtext(cropped_img)

    # Display the extracted text (number plate number)
    number_plate_text = ' '.join([text[1] for text in result])  # Join all detected text
    print(f"Detected Number Plate: {number_plate_text}")

    # Show the image until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# End of code
