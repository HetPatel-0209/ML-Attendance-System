import pandas as pd
import os
import pickle
import glob
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import openpyxl

def load_attendance_data():
    """Load the ML.xlsx attendance file and format it for processing."""
    try:
        # Path to the attendance file
        file_path = "data/attendanceData/2025/IT/sem6/ML.xlsx"
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Attendance file not found: {file_path}")
            return pd.DataFrame(columns=["Name", "Student ID", "Date", "Status"])
        
        # Load the workbook
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active
        
        # Extract data in a format suitable for processing
        attendance_data = []
        
        # Get all column headers (first row)
        headers = [cell.value for cell in sheet[1]]
        
        # Skip processing if sheet is empty or malformed
        if len(headers) < 3:  # Need at least Name, ID, and one date
            return pd.DataFrame(columns=["Name", "Student ID", "Date", "Status"])
        
        # Process each student row
        for row_idx in range(2, sheet.max_row + 1):
            student_name = sheet.cell(row=row_idx, column=1).value
            student_id = sheet.cell(row=row_idx, column=2).value
            
            if not student_name:  # Skip empty rows
                continue
                
            # Process each date column (starting from column 3)
            for col_idx in range(3, sheet.max_column + 1):
                date = headers[col_idx - 1]  # Column headers are 0-indexed in the list
                status = sheet.cell(row=row_idx, column=col_idx).value
                
                if status == 'P':  # Only include 'Present' records
                    attendance_data.append({
                        "Name": student_name,
                        "Student ID": student_id,
                        "Date": date,
                        "Status": "Present"
                    })
        
        # Convert to DataFrame
        return pd.DataFrame(attendance_data)
        
    except Exception as e:
        print(f"Error loading attendance data: {e}")
        return pd.DataFrame(columns=["Name", "Student ID", "Date", "Status"])

def load_student_data():
    """Load student details from CSV."""
    try:
        return pd.read_csv('data/students.csv')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=['student_id', 'name', 'course', 'email', 'phone'])

def generate_training_data(attendance_df, students_df):
    """Generate question-answer pairs for training."""
    qa_pairs = []
    
    # Make sure we have data
    if attendance_df.empty or students_df.empty:
        print("No data available to generate training examples")
        return qa_pairs
    
    # Get unique dates and names
    dates = attendance_df['Date'].unique() if 'Date' in attendance_df.columns else []
    names = attendance_df['Name'].unique() if 'Name' in attendance_df.columns else []
    courses = students_df['course'].unique() if 'course' in students_df.columns else []
    
    # Generate question-answer pairs
    for name in names:
        # Questions about attendance on specific dates
        for date in dates:
            # Was a student present on a specific date?
            present = name in attendance_df[attendance_df['Date'] == date]['Name'].values
            
            question = f"Was {name} present on {date}?"
            answer = f"Yes, {name} was present on {date}." if present else f"No, {name} was not present on {date}."
            qa_pairs.append((question, answer))
            
            # Variation
            question = f"Did {name} attend on {date}?"
            qa_pairs.append((question, answer))
        
        # Questions about student's attendance record
        student_dates = attendance_df[attendance_df['Name'] == name]['Date'].tolist()
        if student_dates:
            question = f"When was {name} present?"
            answer = f"{name} was present on: {', '.join(map(str, student_dates))}"
            qa_pairs.append((question, answer))
            
            # Variation
            question = f"Show me {name}'s attendance record"
            qa_pairs.append((question, answer))
    
    # Questions about all students on a specific date
    for date in dates:
        present_students = attendance_df[attendance_df['Date'] == date]['Name'].tolist()
        if present_students:
            question = f"Who was present on {date}?"
            answer = f"Students present on {date}: {', '.join(present_students)}"
            qa_pairs.append((question, answer))
            
            # Variation
            question = f"Show attendance for {date}"
            qa_pairs.append((question, answer))
            
    # Questions about course attendance
    for course in courses:
        # Get students in this course
        course_students = students_df[students_df['course'] == course]['name'].tolist()
        
        for date in dates:
            # Find who from this course was present on the date
            present_course_students = attendance_df[
                (attendance_df['Date'] == date) & 
                (attendance_df['Name'].isin(course_students))
            ]['Name'].tolist()
            
            if present_course_students:
                question = f"Which {course} students attended on {date}?"
                answer = f"{course} students present on {date}: {', '.join(present_course_students)}"
                qa_pairs.append((question, answer))
    
    # Questions about student details
    for _, student in students_df.iterrows():
        name = student['name']
        question = f"What course is {name} enrolled in?"
        answer = f"{name} is enrolled in {student['course']}"
        qa_pairs.append((question, answer))
        
        question = f"What is {name}'s student ID?"
        answer = f"{name}'s student ID is {student['student_id']}"
        qa_pairs.append((question, answer))
    
    return qa_pairs

def train_retrieval_model(qa_pairs):
    """Train a simple retrieval-based chatbot model."""
    if not qa_pairs:
        print("No training data available")
        return None
    
    questions = [pair[0] for pair in qa_pairs]
    answers = [pair[1] for pair in qa_pairs]
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    
    # Save the model components
    os.makedirs("models", exist_ok=True)
    
    model = {
        'vectorizer': vectorizer,
        'question_vectors': question_vectors,
        'questions': questions,
        'answers': answers
    }
    
    with open('models/chatbot_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Chatbot model trained and saved successfully!")
    return model

def main():
    print("Loading attendance data...")
    attendance_df = load_attendance_data()
    
    print("Loading student data...")
    students_df = load_student_data()
    
    print("Generating question-answer pairs...")
    qa_pairs = generate_training_data(attendance_df, students_df)
    
    print(f"Generated {len(qa_pairs)} question-answer pairs")
    
    print("Training the chatbot model...")
    train_retrieval_model(qa_pairs)

if __name__ == "__main__":
    main()
