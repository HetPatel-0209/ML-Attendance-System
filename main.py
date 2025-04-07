import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import threading
import datetime
from attendance_interface import AttendanceInterface
from create_dirs import create_directory_structure

def run_face_recognition():
    """Run the face recognition app in a separate process"""
    try:
        # Run app.py as a subprocess
        subprocess.Popen([sys.executable, "app.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start face recognition: {str(e)}")

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance Management System")
        self.root.geometry("600x500")
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create Configure tab
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Configure")
        
        # Create Start tab
        self.start_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.start_frame, text="Start Recognition")
        
        # Initialize the interface in the Configure tab
        self.attendance_interface = AttendanceInterface(self.config_frame)
        
        # Create Start Recognition controls
        self.setup_start_tab()
    
    def setup_start_tab(self):
        """Set up the Start Recognition tab"""
        frame = ttk.Frame(self.start_frame, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Current configuration display
        ttk.Label(frame, text="Current Configuration", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky="w")
        
        # Display current configuration values
        self.config_values = tk.StringVar()
        self.update_config_display()
        
        ttk.Label(frame, textvariable=self.config_values).grid(row=1, column=0, columnspan=2, pady=10, sticky="w")
        
        # Start button
        ttk.Button(frame, text="Start Face Recognition", 
                   command=self.start_face_recognition).grid(row=2, column=0, pady=20)
        
        # Refresh config button
        ttk.Button(frame, text="Refresh Configuration", 
                   command=self.update_config_display).grid(row=2, column=1, pady=20)
    
    def update_config_display(self):
        """Update the displayed configuration values"""
        try:
            with open('attendance_config.txt', 'r') as f:
                config_text = ""
                for line in f:
                    key, value = line.strip().split('=', 1)
                    if key != 'filepath':  # Skip filepath to keep display clean
                        config_text += f"{key.capitalize()}: {value}\n"
                self.config_values.set(config_text)
        except FileNotFoundError:
            # Use default values
            current_year = str(datetime.datetime.now().year)
            config_text = f"Year: {current_year}\nProgram: IT\nSemester: sem6\nSubject: ML"
            self.config_values.set(config_text)
    
    def start_face_recognition(self):
        """Start the face recognition application"""
        self.update_config_display()  # Refresh configuration
        
        # Run face recognition in a separate thread to avoid blocking the UI
        thread = threading.Thread(target=run_face_recognition)
        thread.daemon = True
        thread.start()
        
        messagebox.showinfo("Started", "Face recognition started. You can close this dialog.")

if __name__ == "__main__":
    # Create directory structure
    create_directory_structure()
    
    # Create the main window
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop() 