# Face Recognition Attendance System with Chatbot

A comprehensive attendance management system that uses facial recognition to mark attendance and includes a chatbot interface for querying attendance records.

## Features

- **Face Recognition-based Attendance**
  - Automatic face detection and recognition
  - Real-time attendance marking
  - Display of student details during recognition
  - Cooldown system to prevent duplicate entries

- **Student Registration System**
  - GUI interface for student registration
  - Capture student photos
  - Store student details (ID, name, course, email, phone)
  - Face database management

- **Attendance Chatbot**
  - Natural language queries for attendance information
  - Query student attendance records
  - Get course-wise attendance
  - Easy-to-use GUI interface
  - Data refresh capability

## Project Structure

```
├── app.py              # Main facial recognition attendance system
├── register.py         # Student registration GUI
├── chatbot.py         # Attendance query chatbot
├── train_chatbot.py   # Chatbot training module
├── data/
│   ├── students.csv   # Student database
│   └── attendanceData/ # Attendance records
├── faces/             # Student face images
└── models/            # Trained chatbot models
```

## Requirements

- Python 3.10+
- OpenCV
- face_recognition
- dlib
- pandas
- numpy
- scikit-learn
- openpyxl
- tkinter (included with Python)

## Installation

1. Clone the repository
   ```
   git clone https://github.com/HetPatel-0209/Project
   ```
2. Create a virtual environment:
   ```
   python -m venv env
   ```
3. Activate the virtual environment:
   - Windows: `env\Scripts\activate`
   - Unix/MacOS: `source env/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Student Registration
Run the registration system to add new students:
```
python register.py
```
- Fill in student details
- Take a photo using the webcam
- Click "Update Database" to save

### 2. Attendance System
Start the facial recognition attendance system:
```
python app.py
```
- System automatically detects and recognizes faces
- Marks attendance in Excel sheet
- Displays student information in real-time
- Press 'q' or 'ESC' to exit

### 3. Chatbot Interface
Launch the attendance query chatbot:
```
python chatbot.py
```
Example queries:
- "Was [student name] present on [date]?"
- "Who was present on [date]?"
- "What course is [student name] enrolled in?"

## Data Structure

### students.csv
Contains student information:
- Student ID
- Name
- Course
- Email
- Phone

### Attendance Records
- Stored in Excel format
- Path: `data/attendanceData/2025/IT/sem6/ML.xlsx`
- Organized by year, department, and semester
- Marks 'P' for present students

## Notes

- The system uses a 60-second cooldown to prevent duplicate attendance entries
- Face recognition works best with good lighting conditions
- Regular model updates recommended for better chatbot responses

## License

This project is open-source and available under the MIT License.
