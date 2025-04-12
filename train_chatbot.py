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
    """Load all attendance files and format for processing."""
    attendance_data = []
    try:
        # Path to the attendance files
        base_path = "data/attendanceData"
        
        # Walk through all attendance files
        for year_dir in os.listdir(base_path):
            year_path = os.path.join(base_path, year_dir)
            if not os.path.isdir(year_path):
                continue
                
            for branch in os.listdir(year_path):
                branch_path = os.path.join(year_path, branch)
                if not os.path.isdir(branch_path):
                    continue
                    
                for sem in os.listdir(branch_path):
                    sem_path = os.path.join(branch_path, sem)
                    if not os.path.isdir(sem_path):
                        continue
                        
                    for subject_file in glob.glob(os.path.join(sem_path, "*.xlsx")):
                        try:
                            workbook = openpyxl.load_workbook(subject_file)
                            sheet = workbook.active
                            subject = os.path.splitext(os.path.basename(subject_file))[0]
                            
                            headers = [cell.value for cell in sheet[1]]
                            
                            for row_idx in range(2, sheet.max_row + 1):
                                student_name = sheet.cell(row=row_idx, column=1).value
                                student_id = sheet.cell(row=row_idx, column=2).value
                                
                                if not student_name:
                                    continue
                                    
                                # Process each date column
                                for col_idx in range(3, sheet.max_column + 1):
                                    date = headers[col_idx - 1]
                                    status = sheet.cell(row=row_idx, column=col_idx).value
                                    
                                    if status == 'P':
                                        attendance_data.append({
                                            "Name": student_name,
                                            "Student ID": student_id,
                                            "Date": date,
                                            "Status": "Present",
                                            "Year": year_dir,
                                            "Branch": branch,
                                            "Semester": sem,
                                            "Subject": subject
                                        })
                                    elif status == '' or status is None:
                                        attendance_data.append({
                                            "Name": student_name,
                                            "Student ID": student_id,
                                            "Date": date,
                                            "Status": "Absent",
                                            "Year": year_dir,
                                            "Branch": branch,
                                            "Semester": sem,
                                            "Subject": subject
                                        })
                        except Exception as e:
                            print(f"Error processing file {subject_file}: {e}")
                            continue
    except Exception as e:
        print(f"Error loading attendance data: {e}")
    
    if not attendance_data:
        return pd.DataFrame(columns=["Name", "Student ID", "Date", "Status", "Year", "Branch", "Semester", "Subject"])
    return pd.DataFrame(attendance_data)

def load_student_data():
    """Load student details from CSV."""
    try:
        return pd.read_csv('data/students.csv', dtype=str)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=['student_id', 'name', 'course', 'email', 'phone'])

def load_subject_data():
    """Load subject details from CSV."""
    try:
        return pd.read_csv('data/subjects.csv', dtype=str)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()

def calculate_attendance_stats(attendance_df):
    """Calculate attendance statistics for each student."""
    if attendance_df.empty:
        return pd.DataFrame()
        
    stats = attendance_df.groupby(['Name', 'Subject']).agg({
        'Status': lambda x: (x == 'Present').mean()
    }).reset_index()
    
    stats = stats.rename(columns={'Status': 'Attendance_Percentage'})
    stats['Attendance_Percentage'] = stats['Attendance_Percentage'] * 100
    return stats

def generate_training_data(attendance_df, students_df, subjects_df):
    """Generate question-answer pairs for training."""
    qa_pairs = []
    
    # Basic validation
    if attendance_df.empty or students_df.empty:
        print("No data available to generate training examples")
        return qa_pairs
    
    # Calculate attendance statistics
    stats_df = calculate_attendance_stats(attendance_df)
    
    # 1. Student Information Queries
    for _, student in students_df.iterrows():
        # Student ID based queries
        student_id = student['student_id']
        name = student['name']
        
        # Different variations of enrollment/ID queries
        id_questions = [
            f"What is the name of student with enrollment {student_id}?",
            f"Who has enrollment number {student_id}?",
            f"Which student has ID {student_id}?",
            f"Find student with enrollment {student_id}",
            f"Get details of enrollment {student_id}",
            f"Student details for ID {student_id}"
        ]
        
        for question in id_questions:
            answer = f"The student with enrollment {student_id} is {name}"
            qa_pairs.append((question, answer))
        
        # Contact information queries
        contact_questions = [
            f"What is {name}'s phone number?",
            f"Give me contact details of {name}",
            f"What is the contact number of {name}?",
            f"How can I contact {name}?",
            f"What is {name}'s email?",
            f"What are {name}'s contact details?"
        ]
        
        contact_answer = f"Contact details for {name}: Email: {student['email']}, Phone: {student['phone']}"
        for question in contact_questions:
            qa_pairs.append((question, contact_answer))
    
    # 2. Subject Code Queries
    for _, subject in subjects_df.iterrows():
        subject_name = subject['subject_name']
        subject_code = subject['subject_code']
        subject_abbrev = subject['subject_abbrevation']
        
        code_questions = [
            f"What is the subject code for {subject_name}?",
            f"What is the subject code of {subject_name}?",
            f"What is the code for {subject_name}?",
            f"Give me the code for {subject_name}",
            f"Subject code of {subject_name}?",
            # Add abbreviation-based queries
            f"What is the subject code for {subject_abbrev}?",
            f"What is the code for {subject_abbrev}?"
        ]
        
        code_answer = f"The subject code for {subject_name} ({subject_abbrev}) is {subject_code}"
        for question in code_questions:
            qa_pairs.append((question, code_answer))
    
    # 3. Attendance Queries
    for name in students_df['name'].unique():
        student_attendance = attendance_df[attendance_df['Name'] == name]
        
        for subject in subjects_df['subject_name'].unique():
            subject_attendance = student_attendance[student_attendance['Subject'] == subject]
            
            if not subject_attendance.empty:
                # Present dates in subject
                present_dates = subject_attendance[subject_attendance['Status'] == 'Present']['Date'].tolist()
                present_dates_str = ', '.join(sorted(present_dates)) if present_dates else "no dates"
                
                question = f"On which dates was {name} present in {subject}?"
                answer = f"{name} was present in {subject} on: {present_dates_str}"
                qa_pairs.append((question, answer))
                
                # Absent dates in subject
                absent_dates = subject_attendance[subject_attendance['Status'] == 'Absent']['Date'].tolist()
                absent_dates_str = ', '.join(sorted(absent_dates)) if absent_dates else "no dates"
                
                question = f"On which dates was {name} absent in {subject}?"
                answer = f"{name} was absent in {subject} on: {absent_dates_str}"
                qa_pairs.append((question, answer))
                
                # Attendance percentage in subject
                if not stats_df.empty:
                    subject_stats = stats_df[(stats_df['Name'] == name) & (stats_df['Subject'] == subject)]
                    if not subject_stats.empty:
                        percentage = subject_stats.iloc[0]['Attendance_Percentage']
                        question = f"What is {name}'s attendance in {subject}?"
                        answer = f"{name}'s attendance in {subject} is {percentage:.1f}%"
                        qa_pairs.append((question, answer))
                        
                        # Add more variations of the same question
                        question = f"Show me {name}'s attendance record in {subject}"
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
    
    print("Loading subject data...")
    subjects_df = load_subject_data()
    
    print("Generating question-answer pairs...")
    qa_pairs = generate_training_data(attendance_df, students_df, subjects_df)
    
    print(f"Generated {len(qa_pairs)} question-answer pairs")
    
    print("Training the chatbot model...")
    train_retrieval_model(qa_pairs)

if __name__ == "__main__":
    main()
