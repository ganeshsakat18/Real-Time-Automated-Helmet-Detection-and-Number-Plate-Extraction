# Helmet and Number Plate Detection System

## Overview

This project automates helmet detection and number plate extraction to enhance road safety by identifying helmet violations(REAL TIME). Upon detection, violators receive an email with details and an attached image of the violation.

---

## Features

- **Helmet Detection**
  - Detects if riders are wearing helmets using a YOLO model.
  - Marks detections with bounding boxes and labels.
- **Number Plate Extraction**
  - Identifies and extracts vehicle number plates from input images or video frames.
- **Violation Notification**
  - Sends an email with violation details and an attached image.
- **Batch Processing**
  - Processes multiple images in a folder for detection and notification.
- **Real-time Processing**
  - Handles image and video frame processing efficiently.

---

## Workflow

1. **Input**:
   - Accepts images or video frames.
2. **Helmet Detection**:
   - YOLO-based detection for helmets.
   - Labels riders as _"with helmet"_ or _"without helmet"_.
3. **Number Plate Extraction**:
   - Detects number plates.
   - Extracts license plate numbers for further processing.
4. **Violation Handling**:
   - Marks violations if no helmet is detected.
   - Saves processed images in a violation folder.
5. **Email Notification**:
   - Sends email with image and license plate details.
6. **Batch Processing**:
   - Automates detection for multiple images in a folder.

---

## Technologies Used

- **Python**: Core programming language.
- **OpenCV**: Image and video processing.
- **YOLO (You Only Look Once)**: Pre-trained model for detection.
- **EASY OCR**: For extracting text from number plates.
- **smtplib**: For email automation.
- **Flask/FastAPI** (optional): For API integration.

---

## Accomplishments So Far

- **Helmet Detection**:
  - Integrated YOLO model for helmet detection.
  - Added bounding boxes and labels for "with helmet" and "without helmet."
- **Number Plate Detection**:
  - Extracted bounding boxes for number plates.
  - Initial logic for OCR-based text extraction.
- **Email Notification**:
  - Automated email notifications with violation details.
  - Support for folder-based batch email processing.
