import sys
import os

# Add the "scripts" folder to the Python path dynamically
scripts_path = os.path.join(os.path.dirname(__file__), 'scripts')
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

try:
    # Import the show_home_screen function from home_screen module
    from home_screen import show_home_screen
except ImportError as e:
    print(f"Error: Could not import the required module. {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("Starting Helmet & Number Plate Detection System...")
    try:
        # Start the home screen
        show_home_screen()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
