# Face Recognition Attendance System with Chatbot

A comprehensive attendance management system that uses facial recognition technology to automatically mark student attendance and provides a conversational chatbot interface for querying attendance records and student information.

## Features

- **Face Recognition-based Attendance**
  - Real-time face detection and recognition using opencv and face_recognition
  - Automatic attendance marking with timestamp
  - Display of student details during recognition
  - Cooldown system to prevent duplicate entries
  - Unknown face detection with option to register new students on-the-fly

- **Student Registration System**
  - User-friendly GUI interface for student registration
  - Capture student photos directly through webcam
  - Store student details (ID, name, course, email, phone)
  - Support for adding new students to existing database

- **Attendance Management Interface**
  - Configure year, program, semester, and subject
  - Automatic creation of attendance records in Excel format
  - Directory structure organized by academic criteria

- **Attendance Chatbot**
  - Natural language processing for attendance queries
  - Query student attendance records by name or date
  - Get course-wise attendance and statistics
  - Student contact information lookup
  - Data refresh capability for real-time information

- **Main Dashboard**
  - Central hub for accessing all system features
  - Configuration management
  - Quick start for attendance taking

## Project Structure

```
├── app.py                 # Main facial recognition attendance system
├── register.py            # Student registration GUI
├── chatbot.py             # Attendance query chatbot
├── train_chatbot.py       # Chatbot training module
├── attendance_interface.py # Attendance configuration interface
├── main.py                # Main dashboard application
├── create_dirs.py         # Directory structure creation utility
├── attendance_config.txt  # Current attendance configuration
├── data/
│   ├── students.csv       # Student database
│   ├── subjects.csv       # Subject information
│   └── attendanceData/    # Attendance records (organized by year/program/semester)
├── faces/                 # Student face images for recognition
├── models/                # Trained chatbot models
│   └── chatbot_model.pkl  # Serialized chatbot model
└── requirements.txt       # Project dependencies
```

## Technical Components

- **Face Recognition**: Uses face_recognition library with dlib backend for accurate face detection and matching
- **Natural Language Processing**: TF-IDF vectorization and cosine similarity for query understanding
- **Data Management**: Structured CSV and Excel files for student and attendance data
- **User Interface**: Tkinter-based GUIs for all components with modern styling

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
- PIL/Pillow

## Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/face-recognition-attendance.git
   cd face-recognition-attendance
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

5. Create the initial directory structure:
   ```
   python create_dirs.py
   ```

## Usage

### Main Dashboard
Launch the main dashboard to access all system features:
```
python main.py
```
This provides access to both the configuration interface and the attendance system.

### 1. Student Registration
Register new students with their details and face data:
```
python register.py
```
- Fill in student details (ID, name, course, email, phone)
- Take a photo using the webcam
- Click "Update Database" to save the data

### 2. Attendance Management
Configure the attendance settings before taking attendance:
```
python attendance_interface.py
```
- Select the academic year, program, semester, and subject
- Create new subject files if needed

### 3. Attendance System
Start the facial recognition attendance system:
```
python app.py
```
- System automatically detects and recognizes faces
- Marks attendance in Excel sheets organized by year/program/semester/subject
- Displays student information in real-time
- Offers registration for unknown faces
- Press 'q' or 'ESC' to exit

### 4. Chatbot Interface
Query attendance records and student information:
```
python chatbot.py
```
Example queries:
- "Was [student name] present on [date]?"
- "Who was present on [date]?"
- "What is [student name]'s attendance percentage in [subject]?"
- "What is the contact number for [student name]?"
- "What is the subject code for [subject]?"

### 5. Training the Chatbot
Update the chatbot with fresh data and new query patterns:
```
python train_chatbot.py
```
This updates the NLP model with current attendance and student data.

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
- Path: `data/attendanceData/[year]/[program]/[semester]/[subject].xlsx`
- Organized by year, department, and semester
- Marks 'P' for present students

## Implementation Notes

- The system uses face_recognition with a 0.6 distance threshold for face matching
- A 60-second cooldown prevents duplicate attendance entries
- Dynamic directory creation ensures proper data organization
- All components are modular and can run independently or via the main dashboard
- Face recognition works best with good lighting conditions
- Trained chatbot model is serialized to models/chatbot_model.pkl

## Development and Customization

- Add new subjects through the attendance interface
- Modify create_dirs.py to adjust the directory structure
- Edit training data in train_chatbot.py to improve chatbot responses
- Adjust face recognition threshold in app.py for different sensitivity

## License

This project is open-source and available under the MIT License.
