# Attendance Management System

A simple web-based attendance management system that allows you to track student attendance.

## Features

- Input fields for year, date, branch, semester, and subject
- Student list with checkboxes loaded from students.csv file
- Fallback to default student IDs (12202080601001 to 12202080601080) if CSV not available
- Fetch existing attendance data from Excel files
- Submit attendance data to server and save in Excel format
- Select all / Unselect all buttons for convenience

## Setup and Installation

1. Make sure you have Python, Flask, and required packages installed:
   ```
   pip install Flask pandas openpyxl
   ```

2. Navigate to the web directory:
   ```
   cd web
   ```

3. Run the server:
   ```
   python server.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Data Structure

### Student Data
Student information is loaded from `data/students.csv` with the following columns:
- student_id: Unique identifier for each student
- name: Student's name
- course: Student's course/program
- email: Student's email address
- phone: Student's phone number

### Attendance Data
The attendance data is saved in Excel (.xlsx) format with the following structure:
- Base directory: `data/attendanceData/`
- File path: `YYYY/branch/semX/subject/YYYY-MM-DD.xlsx`

The Excel files contain the following columns:
- StudentID: The ID of each student
- Name: The name of the student (if available in students.csv)
- Present: Boolean value indicating whether the student was present (True) or absent (False)
- Date: The date of the attendance record
- Year: The academic year
- Branch: The branch/department
- Semester: The semester number
- Subject: The subject name
- Timestamp: When the attendance was recorded

## Usage

1. Fill in the required information (date, branch, semester, and subject)
2. Click "Fetch Attendance" to load existing data (if any)
3. Mark students as present by checking the corresponding checkboxes
4. Click "Submit Attendance" to save the data

## API Endpoints

- `GET /api/students` - Get list of students from students.csv

- `GET /api/attendance` - Fetch attendance data
  - Query parameters: year, date, branch, sem, subject
  
- `POST /api/attendance` - Save attendance data
  - Request body must contain: year, date, branch, sem, subject, present_students 