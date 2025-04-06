import glob
import face_recognition
import numpy as np
from datetime import datetime
import cv2
import os
import openpyxl
import pandas as pd

now = datetime.now()
dtString = now.strftime("%H:%M")

# Load student details
def load_student_details():
    try:
        return pd.read_csv('data/students.csv')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=['student_id', 'name', 'course', 'email', 'phone'])

# Get student details by name
def get_student_details(name):
    df = load_student_details()
    student = df[df['name'] == name]
    if not student.empty:
        return student.iloc[0].to_dict()
    return None

cap = cv2.VideoCapture(0)
FONT = cv2.FONT_HERSHEY_COMPLEX
images = []
names = []

path = os.path.join('faces', '*.*')
for file in glob.glob(path):
    image = cv2.imread(file)
    a = os.path.basename(file)
    b = os.path.splitext(a)[0]
    names.append(b)
    images.append(image)

def create_or_open_attendance_workbook():
    # Ensure directory exists
    directory = "data/attendanceData/2025/IT/sem6"
    os.makedirs(directory, exist_ok=True)
    
    filepath = f"{directory}/ML.xlsx"
    
    try:
        # Try to load existing workbook
        workbook = openpyxl.load_workbook(filepath)
    except FileNotFoundError:
        # Create new workbook if it doesn't exist
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.cell(row=1, column=1).value = "Student Name"
        sheet.cell(row=1, column=2).value = "Student ID"
        workbook.save(filepath)
    
    sheet = workbook.active
    return workbook, sheet, filepath

def is_date_column_exists(sheet, date_str):
    """Check if column for today's date exists in sheet"""
    for cell in sheet[1]:  # First row
        if cell.value == date_str:
            return cell.column
    return None

def mark_attendance(name):
    student_details = get_student_details(name)
    if not student_details:
        print(f"No details found for {name}")
        return
    
    workbook, sheet, filepath = create_or_open_attendance_workbook()
    
    # Get today's date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Check if student is already in the sheet
    student_row = None
    for row_idx, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
        if row[0] == name:
            student_row = row_idx
            break
    
    # If student not found, add a new row
    if student_row is None:
        student_row = sheet.max_row + 1
        sheet.cell(row=student_row, column=1).value = name
        sheet.cell(row=student_row, column=2).value = student_details.get('student_id', '')
    
    # Find date column or create if not exists
    date_column = is_date_column_exists(sheet, current_date)
    if date_column is None:
        date_column = sheet.max_column + 1
        sheet.cell(row=1, column=date_column).value = current_date
    
    # Mark attendance with 'P'
    current_value = sheet.cell(row=student_row, column=date_column).value
    if current_value != 'P':
        sheet.cell(row=student_row, column=date_column).value = 'P'
        print(f"Attendance marked for {name} on {current_date}")
        workbook.save(filepath)
    else:
        print(f"Attendance already marked for {name} today")

def encoding1(images):
    encode = []
    for img in images:
        unk_encoding = face_recognition.face_encodings(img)[0]
        encode.append(unk_encoding)
    return encode    

# Create a list to track recently recognized faces to avoid duplicate processing
recently_recognized = {}

encodelist = encoding1(images)

# Flag to control the main loop
running = True

def on_close():
    """Handle window close event"""
    global running
    running = False
    cap.release()
    cv2.destroyAllWindows()

# Set window close callback
cv2.namedWindow("Attendance System")

while running:
    ret, frame = cap.read()
    
    # Check if the frame was successfully captured
    if not ret:
        print("Failed to grab frame")
        break
        
    frame1 = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    face_locations = face_recognition.face_locations(frame1)
    curframe_encoding = face_recognition.face_encodings(frame1, face_locations)
    
    for encodeface, facelocation in zip(curframe_encoding, face_locations):
        distance = face_recognition.face_distance(encodelist, encodeface)
        match_index = np.argmin(distance)
        name = names[match_index]
        
        # Check if we've recently recognized this person
        current_time = datetime.now().timestamp()
        if name not in recently_recognized or (current_time - recently_recognized[name]) > 60:  # 60 seconds cooldown
            mark_attendance(name)
            recently_recognized[name] = current_time
        
        # Get student details for display
        student_details = get_student_details(name)
        
        # Draw face rectangle
        x1, y1, x2, y2 = facelocation
        x1, y1, x2, y2 = x1*4, y1*4, x2*4, y2*4
        cv2.rectangle(frame, (y1, x1), (y2, x2), (0,0,255), 3)
        
        # Display student details
        if student_details:
            y_offset = x2 + 30
            cv2.putText(frame, f"Name: {name}", (y1, y_offset), FONT, 0.6, (0,255,0), 1)
            
            if 'student_id' in student_details:
                y_offset += 25
                cv2.putText(frame, f"ID: {student_details['student_id']}", (y1, y_offset), FONT, 0.6, (0,255,0), 1)
            
            if 'course' in student_details:
                y_offset += 25
                cv2.putText(frame, f"Course: {student_details['course']}", (y1, y_offset), FONT, 0.6, (0,255,0), 1)
        else:
            cv2.putText(frame, name, (y2+6, x2-6), FONT, 1, (255,0,255), 2)
    
    cv2.imshow("Attendance System", frame)
    
    # Check for ESC key or q key
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    
    # Check if window was closed
    if cv2.getWindowProperty("Attendance System", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()