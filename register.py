from tkinter import *
from PIL import Image, ImageTk,ImageDraw
import cv2
import face_recognition
import glob
import os
import csv
import pandas as pd

# --- Initialize variables ---
cap = cv2.VideoCapture(0)
images = []
names = []
temp_face_path = "temp_unknown_face.jpg"
using_temp_face = False

# --- Function definitions ---
def snapshot():
    global using_temp_face
    
    # If we're using a loaded face, just save it
    if using_temp_face and os.path.exists(temp_face_path):
        img = Image.open(temp_face_path)
        name = name_var.get()
        os.makedirs('faces', exist_ok=True)
        img.save(os.path.join('faces', name + '.jpg'))
        save_student_details()
        status_label.config(text="Student registered successfully!", fg="green")
        return
        
    # Otherwise capture from camera
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    face_locations = face_recognition.face_locations(cv2image)
    if face_locations:  # Only proceed if at least one face is detected
        for face_location in face_locations:
            top, right, bottom, left = face_location
            im1 = img.crop((left, top, right, bottom))
            name = name_var.get()
            os.makedirs('faces', exist_ok=True)
            im1.save(os.path.join('faces', name + '.jpg'))
            break  # Only save the first detected face
        # Save student details once, after face is saved
        save_student_details()
    else:
        status_label.config(text="No face detected! Please try again.", fg="red")

def save_student_details():
    student_id = id_var.get()
    name = name_var.get()
    course = course_var.get()
    email = email_var.get()
    phone = phone_var.get()
    
    # Validate required fields
    if not student_id or not name:
        status_label.config(text="Student ID and Name are required!", fg="red")
        return
        
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Initialize DataFrame with student columns if file doesn't exist
    if not os.path.exists('data/students.csv'):
        df = pd.DataFrame(columns=['student_id', 'name', 'course', 'email', 'phone'])
    else:
        # Read CSV with student_id as string to prevent type mismatch
        df = pd.read_csv('data/students.csv', dtype={'student_id': str})
    
    # Check if student ID already exists
    if student_id in df['student_id'].values:
        status_label.config(text="Student ID already exists!", fg="red")
        return
    
    # Add new student
    new_student = {
        'student_id': student_id,
        'name': name,
        'course': course,
        'email': email,
        'phone': phone
    }
    
    # Use concat instead of deprecated append
    df = pd.concat([df, pd.DataFrame([new_student])], ignore_index=True)
    
    # Save to CSV
    df.to_csv('data/students.csv', index=False)
    status_label.config(text="Student registered successfully!", fg="green")
    
    # Remove temp file if it exists
    if os.path.exists(temp_face_path):
        os.remove(temp_face_path)

def updatedata():
    path = os.path.join('faces', '*.*')
    for file in glob.glob(path):
        image = cv2.imread(file)
        a = os.path.basename(file)
        b = os.path.splitext(a)[0]
        names.append(b)
        images.append(image)
        print(names)
        
def show_frames():
    global using_temp_face
    
    # If we have a temp face image, display it instead of camera feed
    if os.path.exists(temp_face_path) and not using_temp_face:
        using_temp_face = True
        img = Image.open(temp_face_path)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        status_label.config(text="Unknown face detected! Please enter details to register.", fg="blue")
        return
        
    # Otherwise show live camera feed
    if not using_temp_face:
        cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.after(10, show_frames)
           
def quitapp():
    win.destroy()
    
def return_to_app():
    # Update face database before closing
    updatedata()
    quitapp()
       
# --- Setup GUI ---
win = Tk()
win.title("Student Registration")

# Variables
id_var = StringVar()
name_var = StringVar()
course_var = StringVar()
email_var = StringVar()
phone_var = StringVar()

# Camera frame
label = Label(win)
label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Form frame
form_frame = Frame(win)
form_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")

# Student ID
Label(form_frame, text="Student ID:").grid(row=0, column=0, sticky="w", pady=5)
Entry(form_frame, textvariable=id_var, width=30).grid(row=0, column=1, pady=5)

# Name
Label(form_frame, text="Full Name:").grid(row=1, column=0, sticky="w", pady=5)
Entry(form_frame, textvariable=name_var, width=30).grid(row=1, column=1, pady=5)

# Course
Label(form_frame, text="Course:").grid(row=2, column=0, sticky="w", pady=5)
Entry(form_frame, textvariable=course_var, width=30).grid(row=2, column=1, pady=5)

# Email
Label(form_frame, text="Email:").grid(row=3, column=0, sticky="w", pady=5)
Entry(form_frame, textvariable=email_var, width=30).grid(row=3, column=1, pady=5)

# Phone
Label(form_frame, text="Phone:").grid(row=4, column=0, sticky="w", pady=5)
Entry(form_frame, textvariable=phone_var, width=30).grid(row=4, column=1, pady=5)

# Status label
status_label = Label(form_frame, text="", fg="green")
status_label.grid(row=5, column=0, columnspan=2, pady=10)

# Buttons frame
btn_frame = Frame(win)
btn_frame.grid(row=2, column=0, padx=10, pady=10, sticky="w")

snap_btn = Button(btn_frame, text='Take Photo', command=snapshot, width=15)
snap_btn.grid(row=0, column=0, padx=5)

update_btn = Button(btn_frame, text='Update Database', command=updatedata, width=15)
update_btn.grid(row=0, column=1, padx=5)

done_btn = Button(btn_frame, text='Done', command=return_to_app, width=15)
done_btn.grid(row=0, column=2, padx=5)

quit_btn = Button(btn_frame, text='Quit', command=quitapp, width=15)
quit_btn.grid(row=0, column=3, padx=5)

show_frames()
win.mainloop()
cap.release()