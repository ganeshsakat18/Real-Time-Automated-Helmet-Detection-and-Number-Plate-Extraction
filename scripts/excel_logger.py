import openpyxl
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the Excel file where logs will be saved
LOG_FILE_PATH = "C:/Users/user/Desktop/folder/logs/violations.xlsx"

# Initialize the Excel workbook and sheet
def initialize_excel_log():
    """Initialize the Excel log file and sheet (if not already created)."""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        
        # Create the file if it does not exist
        if not os.path.exists(LOG_FILE_PATH):
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Detection Log"
            ws.append(["Date", "Image/Video Source", "Helmet Status", "License Plate", "Challan Status"])  # Header row
            wb.save(LOG_FILE_PATH)
            logging.info("Excel log file created successfully.")
        else:
            logging.info("Excel log file already exists.")
    except Exception as e:
        logging.error(f"Error initializing Excel log: {e}")

def log_to_excel(source, helmet_detected, plates, challan_status):
    """Log the detection details into the Excel file."""
    try:
        # Load the workbook and select the active sheet
        wb = openpyxl.load_workbook(LOG_FILE_PATH)
        ws = wb.active

        # Get the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare the data to log
        helmet_status = "Detected" if helmet_detected else "Violation"
        plate_numbers = ", ".join(plates) if plates else "N/A"  # Concatenate plate numbers or mark as N/A

        # Append the data to the log
        ws.append([current_time, source, helmet_status, plate_numbers, challan_status])

        # Save the workbook after appending
        wb.save(LOG_FILE_PATH)
        wb.close()  # Ensure the workbook is properly closed
        logging.info("Log entry added to Excel file.")
        return True
    except PermissionError:
        logging.error("Permission denied. Please close the Excel file before logging.")
        return False
    except Exception as e:
        logging.error(f"Error while logging data to Excel: {e}")
        return False

# Initialize the Excel log file when the program starts
initialize_excel_log()

# Example usage:
# log_to_excel("input_images/sample_image.jpg", False, ["ABC1234"], "Sent")
