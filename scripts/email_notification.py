import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Email settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Replace these with your actual email credentials
SENDER_EMAIL = "xyz"  # Your Gmail address
SENDER_PASSWORD = "password"  # Your email password or app-specific password

def send_challan_email(folder_path, plates, recipient_email):
    """Send an email with a challan attachment and violation details."""
    
    # Validate sender's credentials and recipient email
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        logging.error("Error: Email credentials are not set. Please set them in the script.")
        return
    
    if not recipient_email:
        logging.error("Error: Recipient email is missing.")
        return
    
    # Validate folder path
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        logging.error(f"Error: The folder path '{folder_path}' does not exist or is not a directory.")
        return

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    msg['Subject'] = "Helmet Violation Challan"

    # Create the body of the email
    body = (f"Dear User,\n\n"
            f"This is a notification that a helmet violation has been detected. "
            f"The license plate(s) of the vehicle are: {', '.join(plates)}.\n\n"
            "Please find the attached images showing the violation.\n\n"
            "Regards,\nHelmet and Plate Detection System")
    msg.attach(MIMEText(body, 'plain'))

    # Attach all images from the folder
    try:
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            logging.warning(f"No images found in the folder '{folder_path}'. Email will be sent without attachments.")
        else:
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                with open(image_path, 'rb') as img_file:
                    img = MIMEImage(img_file.read())
                    img.add_header('Content-Disposition', 'attachment', filename=image_file)
                    msg.attach(img)
                    logging.info(f"Attached image: {image_file}")
    
    except Exception as e:
        logging.error(f"Error while attaching images: {e}")
        return

    # Try to send the email
    try:
        # Connect to the SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure the connection
        server.login(SENDER_EMAIL, SENDER_PASSWORD)  # Login to the server

        # Send the email
        server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        logging.info(f"Challan email sent successfully to {recipient_email}")

    except smtplib.SMTPException as e:
        logging.error(f"Failed to send email due to SMTP exception: {e}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
    finally:
        server.quit()
