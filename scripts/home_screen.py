import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from detection_logic import process_image, process_video, start_real_time_feed



def select_image():
    """Open file dialog to select an image and process it in a separate thread."""
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*"))
    )
    if image_path:
        def process():
            try:
                process_image(image_path)  # Process the selected image
                messagebox.showinfo("Success", "Image processed successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {e}")
        threading.Thread(target=process, daemon=True).start()

def select_video():
    """Open file dialog to select a video and process it in a separate thread."""
    video_path = filedialog.askopenfilename(
        title="Select a Video",
        filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
    )
    if video_path:
        def process():
            try:
                process_video(video_path)  # Process the selected video
                messagebox.showinfo("Success", "Video processed successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process video: {e}")
        threading.Thread(target=process, daemon=True).start()

def start_feed():
    """Start the real-time camera feed in a separate thread."""
    def process():
        try:
            start_real_time_feed()  # Start the real-time feed for detection
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start real-time feed: {e}")
    threading.Thread(target=process, daemon=True).start()

def show_home_screen():
    """Show the main home screen with options for processing images, videos, or real-time feed."""
    app = tk.Tk()  # Create main window
    app.title("Helmet & Number Plate Detection System")
    app.geometry("400x400")  # Set window size

    # Title Label
    title_label = tk.Label(app, text="Helmet & Number Plate Detection", font=("Arial", 16, "bold"))
    title_label.pack(pady=20)

    # Process Image Button
    btn_image = tk.Button(
        app, text="Process Image", command=select_image, font=("Arial", 12),
        width=20, height=2, bg="#4CAF50", fg="white", relief="flat"
    )
    btn_image.pack(pady=10)

    # Process Video Button
    btn_video = tk.Button(
        app, text="Process Video", command=select_video, font=("Arial", 12),
        width=20, height=2, bg="#4CAF50", fg="white", relief="flat"
    )
    btn_video.pack(pady=10)

    # Start Real-Time Feed Button
    btn_real_time = tk.Button(
        app, text="Start Real-Time Feed", command=start_feed, font=("Arial", 12),
        width=20, height=2, bg="#4CAF50", fg="white", relief="flat"
    )
    btn_real_time.pack(pady=10)

    # # Dashboard Button
    # btn_dashboard = tk.Button(
    #     app, text="Open Dashboard", command=open_dashboard, font=("Arial", 12),
    #     width=20, height=2, bg="#FF9800", fg="white", relief="flat"
    # )
    # btn_dashboard.pack(pady=10)

    # Exit Button
    btn_exit = tk.Button(
        app, text="Exit", command=app.quit, font=("Arial", 12),
        width=20, height=2, bg="#f44336", fg="white", relief="flat"
    )
    btn_exit.pack(pady=10)

    # Start the Tkinter event loop to display the home screen
    app.mainloop()

# Debugging: Print current directory
import os
print("Current Directory:", os.getcwd())

# Run the Home Screen
if __name__ == "__main__":
    show_home_screen()
