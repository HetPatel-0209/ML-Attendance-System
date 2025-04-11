from flask import Flask, request, jsonify, send_from_directory
import os
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import csv

app = Flask(__name__, static_folder='.', static_url_path='')

# Data directory structure: data/attendanceData/YYYY/branch/sem/subject/YYYY-MM-DD.xlsx

def get_attendance_path(year, date, branch, sem, subject):
    """Generate the path to the attendance file"""
    base_dir = os.path.join('..', 'data', 'attendanceData', str(year), str(branch), f"sem{sem}", str(subject))
    file_name = f"{date}.xlsx"
    file_path = os.path.join(base_dir, file_name)
    return base_dir, file_path

def get_student_data():
    """Read student data from CSV file"""
    students = []
    try:
        csv_path = os.path.join('..', 'data', 'students.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    students.append(row)
        return students
    except Exception as e:
        print(f"Error reading student data: {str(e)}")
        return []

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/api/students', methods=['GET'])
def get_students():
    """Get list of students filtered by branch"""
    branch = request.args.get('branch')
    students = get_student_data()
    if branch:
        students = [s for s in students if s.get('course') == branch]
    return jsonify(students)

@app.route('/api/subjects', methods=['GET'])
def get_subjects():
    """Get list of subjects for a branch and semester"""
    year = request.args.get('year')
    branch = request.args.get('branch')
    sem = request.args.get('sem')
    
    if not all([year, branch, sem]):
        return jsonify({'error': 'Missing required parameters'}), 400
        
    # Get path to semester directory
    path = os.path.join('..', 'data', 'attendanceData', str(year), str(branch), f"sem{sem}")
    
    # Get all Excel files (subjects) in the directory
    if os.path.exists(path):
        subjects = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.xlsx')]
        return jsonify(subjects)
    
    return jsonify([])

@app.route('/api/attendance', methods=['GET', 'POST'])
def handle_attendance():
    if request.method == 'GET':
        # Get parameters from request
        year = request.args.get('year')
        date = request.args.get('date')
        branch = request.args.get('branch')
        sem = request.args.get('sem')
        subject = request.args.get('subject')
        
        # Validate parameters
        if not all([year, date, branch, sem, subject]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Get file path - note we're getting the subject file, not a date-specific file
        file_path = os.path.join('..', 'data', 'attendanceData', str(year), str(branch), f"sem{sem}", f"{subject}.xlsx")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'message': 'No attendance data found', 'present_students': []}), 200
        
        try:
            # Read the Excel file
            df = pd.read_excel(file_path)
            present_students = []
            
            # Check if the date column exists
            if date in df.columns:
                # Get enrollment numbers of students marked as 'P' for this date
                # Map student IDs to match the format in students.csv
                present_students = df[df[date] == 'P']['Enrollment'].astype(str).tolist()
            
            attendance_data = {
                'date': date,
                'year': year,
                'branch': branch,
                'sem': sem,
                'subject': subject,
                'present_students': present_students
            }
            
            return jsonify(attendance_data), 200
            
        except Exception as e:
            return jsonify({'error': f'Error reading attendance data: {str(e)}'}), 500
    
    elif request.method == 'POST':
        # Get data from request body
        data = request.json
        
        # Validate data
        required_fields = ['year', 'date', 'branch', 'sem', 'subject', 'present_students']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields in request body'}), 400
        
        try:
            # Format data for saving
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            web_test_data = {
                'timestamp': timestamp,
                'year': data['year'],
                'date': data['date'],
                'branch': data['branch'],
                'semester': data['sem'],
                'subject': data['subject'],
                'present_students': data['present_students']
            }
            
            # Save to webTest folder as text file
            os.makedirs('../data/webTest', exist_ok=True)
            test_file_path = os.path.join('..', 'data', 'webTest', f'attendance_submission_{timestamp}.txt')
            
            with open(test_file_path, 'w') as f:
                for key, value in web_test_data.items():
                    f.write(f"{key}: {value}\n")
            
            # Also save to the main attendance system
            main_path = os.path.join('..', 'data', 'attendanceData', str(data['year']), 
                                   str(data['branch']), f"sem{data['sem']}", f"{data['subject']}.xlsx")
            
            try:
                # If file exists, read it; otherwise create new DataFrame
                if os.path.exists(main_path):
                    df = pd.read_excel(main_path)
                else:
                    # Get students from CSV
                    students_df = pd.read_csv('../data/students.csv')
                    df = pd.DataFrame({
                        'Name': students_df['name'],
                        'Enrollment': students_df['student_id']
                    })
                
                # Add or update the date column with attendance
                df[data['date']] = df['Enrollment'].apply(
                    lambda x: 'P' if str(x) in data['present_students'] else ''
                )
                
                # Ensure directory exists and save
                os.makedirs(os.path.dirname(main_path), exist_ok=True)
                df.to_excel(main_path, index=False)
                
            except Exception as e:
                print(f"Warning: Could not save to main attendance system: {str(e)}")
                # Continue since we at least saved the test data
            
            return jsonify({'message': 'Attendance data saved successfully'}), 200
            
        except Exception as e:
            return jsonify({'error': f'Error saving attendance data: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)