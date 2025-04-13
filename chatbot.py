import pickle
import tkinter as tk
from tkinter import scrolledtext, ttk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess
import re
from train_chatbot import load_attendance_data, load_student_data, load_subject_data, calculate_attendance_stats
from fuzzywuzzy import fuzz
# Update imports to use langchain-community packages
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

class AttendanceChatbot:
    def __init__(self):
        self.legacy_model = None
        try:
            self.legacy_model = self.load_legacy_model()
        except:
            print("Legacy model not found. Will use RAG pipeline only.")
        
        self.refresh_data()
        self.setup_rag_pipeline()
        
    def load_legacy_model(self):
        """Load the trained chatbot model."""
        try:
            with open('models/chatbot_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            print("Model file not found. Please run train_chatbot.py first.")
            return None
    
    def refresh_data(self):
        try:
            self.attendance_df = load_attendance_data()
            self.students_df = load_student_data()
            self.subjects_df = load_subject_data()
            self.stats_df = calculate_attendance_stats(self.attendance_df)
        except Exception as e:
            print(f"Error loading data: {e}")
        # Initialize empty DataFrames to prevent crashes
            self.attendance_df = pd.DataFrame()
            self.students_df = pd.DataFrame()
            self.subjects_df = pd.DataFrame()
            self.stats_df = pd.DataFrame()
        
    def setup_rag_pipeline(self):
        """Set up the RAG pipeline using LangChain."""
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector stores from our dataframes
        self.create_vector_stores()
        
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama3-8b-8192",
            temperature=0.1,
            max_tokens=512
        )
        
        # Create retrieval chain
        self.setup_retrieval_chain()
    
    def create_vector_stores(self):
        """Create vector stores from dataframes for RAG."""
        all_docs = []
        
        # Create a copy of the dataframes with all columns converted to strings
        students_df_str = self.students_df.astype(str)
        subjects_df_str = self.subjects_df.astype(str)
        attendance_df_str = self.attendance_df.astype(str)
        stats_df_str = self.stats_df.astype(str)
        
        # Process students data with multiple content columns
        for column in ["name", "student_id", "course", "email", "phone"]:
            if column in students_df_str.columns:
                try:
                    loader = DataFrameLoader(students_df_str, page_content_column=column)
                    all_docs.extend(loader.load())
                except Exception as e:
                    print(f"Error loading student data for column {column}: {e}")
        
        # Process subjects data - include all relevant columns
        for column in ["subject_name", "subject_abbrevation", "subject_code", "type", "credits", "semester", "branch"]:
            if column in subjects_df_str.columns:
                try:
                    loader = DataFrameLoader(subjects_df_str, page_content_column=column)
                    all_docs.extend(loader.load())
                except Exception as e:
                    print(f"Error loading subject data for column {column}: {e}")
            if "credits" in subjects_df_str.columns:
                loader = DataFrameLoader(subjects_df_str, page_content_column="credits")
                all_docs.extend(loader.load())
        
        # Process attendance data
        try:
            attendance_loader = DataFrameLoader(attendance_df_str, page_content_column="Name")
            attendance_docs = attendance_loader.load()
            all_docs.extend(attendance_docs)
            
            # Also add entries for each subject, date, status combination
            for key in ["Subject", "Date", "Status", "Year", "Branch", "Semester"]:
                if key in attendance_df_str.columns:
                    loader = DataFrameLoader(attendance_df_str, page_content_column=key)
                    all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading attendance data: {e}")
        
        # Process stats data
        try:
            stats_loader = DataFrameLoader(stats_df_str, page_content_column="Name")
            stats_docs = stats_loader.load()
            all_docs.extend(stats_docs)
            
            # Also add attendance percentage data
            if "Attendance_Percentage" in stats_df_str.columns:
                stats_pct_loader = DataFrameLoader(stats_df_str, page_content_column="Attendance_Percentage")
                all_docs.extend(stats_pct_loader.load())
        except Exception as e:
            print(f"Error loading stats data: {e}")
        
        # Split documents if they are too long
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(all_docs)
        
        # Create the vector store
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
    
    def setup_retrieval_chain(self):
        """Set up the retrieval chain for answering questions."""
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5})
        )

    def get_real_time_answer(self, question):
        """Generate answers for real-time queries that need fresh data."""
        question_lower = question.lower()
        
        # Debug log the original question
        print(f"ORIGINAL QUERY: '{question}'")
        
        # Direct handling for missed attendance on date queries
        if ("missed" in question_lower and "attendance" in question_lower and "on" in question_lower) or \
           ("absent" in question_lower and "on" in question_lower) or \
           ("students" in question_lower and "absent" in question_lower) or \
           ("which students missed" in question_lower):
            # Extract date using regex
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',                   # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',                   # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}'                    # DD-MM-YYYY
            ]
            
            for pattern in date_patterns:
                date_matches = re.findall(pattern, question)
                if date_matches:
                    date = date_matches[0]
                    print(f"Found date for absence query: {date}")
                    return self._handle_students_missing_on_date(date)
                    
        # Process quoted text directly for certain types of queries
        quoted_subject = None
        if '"' in question:
            # Extract text within double quotes
            quoted_matches = re.findall(r'"([^"]*)"', question)
            if quoted_matches:
                quoted_subject = quoted_matches[0].strip()
                print(f"Found quoted subject: '{quoted_subject}'")
        elif "'" in question:
            # Extract text within single quotes
            quoted_matches = re.findall(r"'([^']*)'", question)
            if quoted_matches:
                quoted_subject = quoted_matches[0].strip()
                print(f"Found quoted subject: '{quoted_subject}'")
                
        # Extract useful entities from the question
        result = self._extract_entities(question_lower)
        student_name = result.get('student_name')
        enrollment_id = result.get('enrollment_id')
        subject = result.get('subject')
        query_type = result.get('query_type')
        date = result.get('date')
        semester = result.get('semester')  # Get extracted semester
        branch = result.get('branch')      # Get extracted branch
        month_year = result.get('month_year')  # Get extracted month-year
        
        # For cases where quotes are used, prioritize quoted text over extracted subject
        if quoted_subject:
            # Directly check if quoted text appears in subject names in our database
            for index, row in self.subjects_df.iterrows():
                if quoted_subject.lower() in row['subject_name'].lower():
                    subject = quoted_subject
                    break
        
        # Debug print to see extracted entities
        print(f"Extracted entities: {result}")
        print(f"Using subject: {subject}, Quoted subject: {quoted_subject}")
        
        # Check for general patterns to determine query intent
        # This helps identify what the user is asking for, regardless of entity extraction
        is_attendance_query = any(phrase in question_lower for phrase in [
            "attendance", "present", "absent", "percentage", "percentages", "stats", 
            "how many classes", "missed class", "attendace", "attend", "miss", "missed"  # Common misspelling
        ])
        
        is_course_code_query = any(phrase in question_lower for phrase in [
            "course code", "subject code", "code of", "code for", "course code of", "subject code of"
        ])
        
        is_credit_query = any(phrase in question_lower for phrase in [
            "credit distribution", "credits for semester", "credits in semester", 
            "total credits", "semester credits", "how many credits"
        ])
        
        is_elective_query = any(phrase in question_lower for phrase in [
            "elective", "open elective", "professional elective", "is it an elective",
            "what type of subject", "subject type", "type of elective"
        ])
        
        print(f"Is attendance query: {is_attendance_query}")
        print(f"Is course code query: {is_course_code_query}")
        print(f"Is credit query: {is_credit_query}")
        print(f"Is elective query: {is_elective_query}")
        
        # HANDLE CREDIT DISTRIBUTION QUERIES - Higher priority
        if is_credit_query and semester:
            print(f"Processing credit distribution query for semester {semester}")
            return self._handle_semester_credits_query(semester)
        
        # HANDLE ELECTIVE TYPE QUERIES
        if is_elective_query:
            # For elective queries, prefer the quoted subject over extracted subject
            search_subject = quoted_subject if quoted_subject else subject
            if search_subject:
                print(f"Processing elective type query for '{search_subject}'")
                return self._handle_subject_type_query(search_subject)
            
        # Check if query mentions 'attendance' and a subject, prioritize it as an attendance query
        if is_attendance_query and subject:
            print(f"Prioritizing as attendance query for subject: {subject}")
            query_type = 'attendance'
            
        # Find the best match for student name if one is found
        student_match = None
        if student_name:
            student_match = self._find_best_name_match(student_name, self.students_df['name'].unique())
            print(f"Best student match: {student_match}")
        
        # Try to find partial name matches in the question for names not directly extracted
        if not student_match:
            for name in self.students_df['name'].unique():
                first_name = name.split()[0]
                if first_name.lower() in question_lower:
                    student_match = name
                    print(f"Found partial name match: {student_match}")
                    break
        
        # HANDLE ABSENCE QUERIES - Check for students missing classes
        if student_match and subject and (
            "miss" in question_lower or 
            "missed" in question_lower or 
            "absent" in question_lower or 
            "absences" in question_lower or
            ("which" in question_lower and "dates" in question_lower)
        ):
            print(f"Processing absence query for {student_match} in {subject}")
            return self._handle_student_absence_query(student_match, subject)
            
        # HANDLE MONTH-YEAR ATTENDANCE QUERIES
        if month_year and subject and student_match:
            print(f"Processing month-year attendance for {month_year} in {subject}")
            # Use a dedicated method to handle attendance over a month
            return self._handle_month_year_attendance(month_year, subject, student_match)
        
        # HANDLE DATE-SPECIFIC ATTENDANCE QUERIES - Check this first
        if date and subject and (
            "on" in question_lower or 
            "for date" in question_lower or 
            "for the date" in question_lower or
            ("list" in question_lower and "records" in question_lower) or
            ("attendance" in question_lower and "records" in question_lower) or
            (any(word in question_lower for word in ["record", "records", "status", "list"]) and "on" in question_lower)
        ):
            print(f"Processing date-specific attendance for {date} in {subject}")
            # Use a dedicated method to handle this special case
            return self._handle_date_specific_attendance(date, subject, student_match)
        
        # HANDLE LISTING STUDENTS QUERIES
        if ('list all students' in question_lower or 'show all students' in question_lower or 
            'list students' in question_lower):
            branch = result.get('branch')
            return self._handle_student_list_query(branch)
        
        # HANDLE SEMESTER SUBJECTS QUERIES
        if query_type == 'semester_subjects' or (
            ('subjects' in question_lower or 'courses' in question_lower) and 
            ('offered' in question_lower or 'in semester' in question_lower or 'in sem' in question_lower)):
            
            # Extract semester and branch from the query
            if semester and branch:
                return self._handle_subject_list_query(semester, branch)
            elif semester:
                return self._handle_subject_list_query(semester)
        
        # HANDLE CLASS COUNT QUERIES
        if query_type == 'class_count' and subject:
            return self._handle_class_count_query(subject)
            
        # HANDLE ZERO CREDIT QUERIES
        if query_type == 'zero_credit':
            if semester:
                return self._handle_zero_credit_query(semester)
            else:
                # Try to extract semester from question directly
                sem_pattern = r'(?:sem(?:ester)?|sem\.)\s*(\d+)'
                sem_matches = re.findall(sem_pattern, question_lower)
                if sem_matches:
                    extracted_semester = sem_matches[0]
                    return self._handle_zero_credit_query(extracted_semester)
                else:
                    sem_numeric = re.findall(r'(?:in|for|semester)\s+(\d)(?:\b|st|nd|rd|th)', question_lower)
                    if sem_numeric:
                        extracted_semester = sem_numeric[0]
                        return self._handle_zero_credit_query(extracted_semester)
                    return "Please specify a semester to check for zero-credit subjects"
        
        # HANDLE ELECTIVE COURSES QUERIES
        if query_type == 'electives' and semester:
            elective_type = subject if subject else None
            return self._handle_elective_queries(semester, elective_type)
            
        # Check for professional elective query patterns
        if "professional elective" in question_lower or "pe subject" in question_lower or "pe subjects" in question_lower:
            # Extract semester if mentioned
            if semester:
                return self._handle_elective_queries(semester, "Professional Elective")
            else:
                # Look for semester in the query
                sem_pattern = r'(?:sem(?:ester)?|sem\.)\s*(\d+)'
                sem_matches = re.findall(sem_pattern, question)
                if sem_matches:
                    semester = sem_matches[0]
                    return self._handle_elective_queries(semester, "Professional Elective")
                else:
                    pe_count = len(self.subjects_df[self.subjects_df['subject_type'] == 'Professional Elective'])
                    return f"There are {pe_count} Professional Elective subjects across all semesters"
        
        # HANDLE ATTENDANCE LIST QUERIES
        if query_type == 'attendance_list' and student_match:
            return self._handle_student_attendance_list(student_match)
            
        # HANDLE LOWEST/HIGHEST ATTENDANCE QUERIES
        if query_type in ['lowest_attendance', 'highest_attendance'] and student_match:
            return self._handle_extreme_attendance_query(student_name, query_type)
        
        # COURSE CODE QUERY HANDLING - Process first if detected
        if is_course_code_query:
            print(f"Processing course code query for: {subject}")
            
            # Handle exact matches in quotes
            quoted_subject = None
            if '"' in question:
                # Extract text within double quotes
                quoted_matches = re.findall(r'"([^"]*)"', question)
                if quoted_matches:
                    quoted_subject = quoted_matches[0].strip()
            elif "'" in question:
                # Extract text within single quotes
                quoted_matches = re.findall(r"'([^']*)'", question)
                if quoted_matches:
                    quoted_subject = quoted_matches[0].strip()
                
            if quoted_subject:
                print(f"Found quoted subject: '{quoted_subject}'")
                
                # Try exact match on subject name
                exact_match = self.subjects_df[
                    self.subjects_df['subject_name'].str.lower() == quoted_subject.lower()
                ]
                
                if not exact_match.empty:
                    subject_info = exact_match.iloc[0]
                    return f"The subject code for {subject_info['subject_name']} ({subject_info['subject_abbrevation']}) is {subject_info['subject_code']}"
                    
                # If no exact match, try partial match
                partial_matches = self.subjects_df[
                    self.subjects_df['subject_name'].str.lower().str.contains(quoted_subject.lower())
                ]
                
                if not partial_matches.empty:
                    subject_info = partial_matches.iloc[0]
                    return f"The subject code for {subject_info['subject_name']} ({subject_info['subject_abbrevation']}) is {subject_info['subject_code']}"
            
            # Direct handling for specific well-known abbreviations
            if subject and subject.upper() in ['PWP', 'ML', 'AI', 'DBMS', 'OS', 'IOT', 'INS', 'SCM', 'LAVCODE']:
                subject_info = self.subjects_df[self.subjects_df['subject_abbrevation'] == subject.upper()]
                
                if not subject_info.empty:
                    found_subject = subject_info.iloc[0]
                    return f"The subject code for {found_subject['subject_name']} ({found_subject['subject_abbrevation']}) is {found_subject['subject_code']}"
            
            # For queries about course code with no specific subject mentioned
            # Try to extract subject from the question text itself
            if not subject:
                for abbrev in self.subjects_df['subject_abbrevation'].unique():
                    if abbrev.lower() in question_lower:
                        subject = abbrev
                        print(f"Found subject mention in question: {subject}")
                        break
                
                for subj_name in self.subjects_df['subject_name'].unique():
                    if subj_name.lower() in question_lower:
                        subject = subj_name
                        print(f"Found subject name in question: {subject}")
                        break
                        
            # Now try to find the subject by name or abbreviation
            if subject:
                # Try exact match on abbreviation (case-insensitive)
                exact_match = self.subjects_df[
                    self.subjects_df['subject_abbrevation'].str.lower() == subject.lower()
                ]
                
                if not exact_match.empty:
                    subject_info = exact_match.iloc[0]
                    return f"The subject code for {subject_info['subject_name']} ({subject_info['subject_abbrevation']}) is {subject_info['subject_code']}"
                
                # Try exact match on subject name (case-insensitive)
                exact_name_match = self.subjects_df[
                    self.subjects_df['subject_name'].str.lower() == subject.lower()
                ]
                
                if not exact_name_match.empty:
                    subject_info = exact_name_match.iloc[0]
                    return f"The subject code for {subject_info['subject_name']} ({subject_info['subject_abbrevation']}) is {subject_info['subject_code']}"
                
                # Try partial match if exact match failed
                subject_matches = self.subjects_df[
                    self.subjects_df['subject_name'].str.lower().str.contains(subject.lower()) | 
                    self.subjects_df['subject_abbrevation'].str.lower().str.contains(subject.lower())
                ]
                
                if not subject_matches.empty:
                    subject_info = subject_matches.iloc[0]
                    return f"The subject code for {subject_info['subject_name']} ({subject_info['subject_abbrevation']}) is {subject_info['subject_code']}"
        
        # ATTENDANCE QUERIES
        if is_attendance_query:
            # Handle general attendance query for a student (without specific subject)
            if student_match and not subject:
                print(f"Checking {student_match}'s overall attendance")
                
                student_stats = self.stats_df[self.stats_df['Name'] == student_match]
                if not student_stats.empty and 'Attendance_Percentage' in student_stats.columns:
                    avg_percentage = student_stats['Attendance_Percentage'].mean()
                    return f"{student_match}'s overall attendance is {avg_percentage:.1f}%"
                else:
                    # If no stats available in stats_df, calculate from raw attendance data
                    student_attendance = self.attendance_df[self.attendance_df['Name'] == student_match]
                    if not student_attendance.empty:
                        avg_percentage = (student_attendance['Status'] == 'Present').mean() * 100
                        return f"{student_match}'s overall attendance is {avg_percentage:.1f}%"
                    else:
                        # No attendance records found for this student
                        return f"No attendance records found for {student_match}"
            
            # Handle specific student attendance in all subjects
            if student_match and "all subjects" in question_lower:
                print(f"Checking {student_match}'s attendance across all subjects")
                response = f"{student_match}'s attendance percentages across subjects:\n"
                
                # Get attendance for each subject
                for subject_name in self.attendance_df['Subject'].unique():
                    subject_attendance = self.attendance_df[
                        (self.attendance_df['Name'] == student_match) &
                        (self.attendance_df['Subject'] == subject_name)
                    ]
                    
                    if not subject_attendance.empty:
                        percentage = (subject_attendance['Status'] == 'Present').mean() * 100
                        response += f"- {subject_name}: {percentage:.1f}%\n"
                
                return response
            
            # Date and subject attendance query - handle both together
            if date and student_match and subject:
                print(f"Checking if {student_match} was present on {date} in {subject}")
                
                # Filter for this specific date, student, and subject
                attendance = self.attendance_df[
                    (self.attendance_df['Name'] == student_match) &
                    (self.attendance_df['Date'] == date) &
                    (self.attendance_df['Subject'].str.lower() == subject.lower())  # Case-insensitive
                ]
                
                if not attendance.empty:
                    status = attendance.iloc[0]['Status']
                    return f"{student_match} was {status} on {date} in {subject.upper()}"
                else:
                    # Try less strict subject matching
                    attendance = self.attendance_df[
                        (self.attendance_df['Name'] == student_match) &
                        (self.attendance_df['Date'] == date)
                    ]
                    
                    if not attendance.empty:
                        # We have attendance for this student on this date, but not for this subject
                        attended_subjects = attendance['Subject'].tolist()
                        statuses = attendance['Status'].tolist()
                        
                        subject_statuses = [f"{subj}: {status}" for subj, status in zip(attended_subjects, statuses)]
                        return f"{student_match} has the following attendance on {date}: {', '.join(subject_statuses)}"
                    
                    return f"No attendance record found for {student_match} on {date} in {subject.upper()}"

            # Handle specific student attendance in a subject (overall, not date-specific)
            if student_match and subject and not date:
                print(f"Checking {student_match}'s overall attendance in {subject}")
                
                # Get the subject attendance data
                stats_df = self._calculate_subject_attendance(subject)
                
                if stats_df is not None:
                    # Filter for just this student
                    student_stats = stats_df[stats_df['Name'] == student_match]
                    
                    if not student_stats.empty:
                        attendance_pct = student_stats.iloc[0]['Status']
                        return f"{student_match}'s attendance in {subject.upper()} is {attendance_pct:.1f}%"
                    return f"No attendance records found for {student_match} in {subject.upper()}"
            
            # Handle attendance percentage queries for all students
            if subject and not student_match and not date:
                print(f"Processing attendance percentage query for subject: {subject}")
                stats_df = self._calculate_subject_attendance(subject)
                if stats_df is not None:
                    response = f"Attendance percentages in {subject.upper()}:\n"
                    for _, row in stats_df.iterrows():
                        response += f"- {row['Name']}: {row['Status']:.1f}%\n"
                    return response
                return f"No attendance data available for {subject}"
        
        # Date-based attendance queries without specific mention of "attendance"
        if date and student_match:
            attendance = self.attendance_df[
                (self.attendance_df['Name'] == student_match) &
                (self.attendance_df['Date'] == date)
            ]
            if not attendance.empty:
                status = attendance.iloc[0]['Status']
                subject_info = attendance.iloc[0].get('Subject', 'all subjects')
                return f"{student_match} was {status} on {date} for {subject_info}"
            return f"No attendance record found for {student_match} on {date}"

        # Handle enrollment ID lookup
        if enrollment_id:
            student = self.students_df[self.students_df['student_id'] == enrollment_id]
            if not student.empty:
                return f"Enrollment {enrollment_id} belongs to {student.iloc[0]['name']}"
            else:
                return f"No student found with enrollment ID {enrollment_id}"
        
        # Handle student name-based queries
        if student_name or student_match:
            # Use student_match if available, otherwise try to find match by name
            if not student_match and student_name:
                student_match = self._find_best_name_match(student_name, self.students_df['name'].unique())
                
            if student_match:
                student = self.students_df[self.students_df['name'] == student_match]
                
                if not student.empty:
                    student_info = student.iloc[0]
                    
                    # Handle specific query types
                    if query_type == 'enrollment':
                        return f"The enrollment number for {student_match} is {student_info['student_id']}"
                    
                    elif query_type == 'contact':
                        return f"Contact details for {student_match}: Email: {student_info['email']}, Phone: {student_info['phone']}"
                    
                    elif query_type == 'email':
                        return f"{student_match}'s email is {student_info['email']}"
                    
                    elif query_type == 'phone':
                        return f"{student_match}'s phone number is {student_info['phone']}"
                    
                    # If asking about course/branch
                    elif any(word in question_lower for word in ['course', 'branch', 'enrolled in', 'studying']):
                        return f"{student_match} is enrolled in {student_info['course']}"
                    
                    # Add more query types as needed
                    
                    # If no specific query type is identified but we have a student match
                    if not query_type and 'enrollment' in question_lower:
                        return f"The enrollment number for {student_match} is {student_info['student_id']}"
        
        # Check for 'full form' queries about subjects
        if 'full form' in question_lower and subject:
            subject_matches = self.subjects_df[
                (self.subjects_df['subject_abbrevation'].str.lower() == subject.lower()) |
                (self.subjects_df['subject_name'].str.lower().str.contains(subject.lower()))
            ]

            if not subject_matches.empty:
                subject_info = subject_matches.iloc[0]
                return f"The full form of {subject_info['subject_abbrevation']} is {subject_info['subject_name']}"
        
        # Handle subject-related queries - MOVED AFTER attendance percentage handling
        if subject and not query_type and not is_attendance_query and not is_course_code_query:
            # Try to find subject by name or abbreviation
            subject_matches = self.subjects_df[
                (self.subjects_df['subject_abbrevation'].str.lower() == subject.lower()) |
                (self.subjects_df['subject_name'].str.lower() == subject.lower())
            ]
            
            if not subject_matches.empty:
                # Check if the query is about attendance first
                if any(word in question_lower for word in ["attendance", "present", "absent"]):
                    # This is likely about attendance, not subject code
                    stats_df = self._calculate_subject_attendance(subject)
                    if stats_df is not None:
                        response = f"Attendance percentages in {subject.upper()}:\n"
                        for _, row in stats_df.iterrows():
                            response += f"- {row['Name']}: {row['Status']:.1f}%\n"
                        return response
                
                # If not about attendance, return subject details
                subject_info = subject_matches.iloc[0]
                return f"Details for {subject_info['subject_name']} ({subject_info['subject_abbrevation']}):\n" \
                       f"- Subject Code: {subject_info['subject_code']}\n" \
                       f"- Credits: {subject_info['credits']}\n" \
                       f"- Type: {subject_info['subject_type']}"
        
        # Handle subject code queries explicitly
        if query_type == 'subject_code' and subject:
            subject_matches = self.subjects_df[
                self.subjects_df['subject_name'].str.lower().str.contains(subject) | 
                self.subjects_df['subject_abbrevation'].str.lower().str.contains(subject)
            ]
            
            if not subject_matches.empty:
                subject_info = subject_matches.iloc[0]
                return f"The subject code for {subject_info['subject_name']} ({subject_info['subject_abbrevation']}) is {subject_info['subject_code']}"
        
        # Handle subject-wide attendance percentage
        if any(phrase in question_lower for phrase in ["attendance percentage", "attendance percentages", "attendance stats"]) and subject:
            stats_df = self._calculate_subject_attendance(subject)
            if stats_df is not None:
                response = f"Attendance percentages in {subject.upper()}:\n"
                for _, row in stats_df.iterrows():
                    response += f"- {row['Name']}: {row['Status']:.1f}%\n"
                return response
            return f"No attendance data available for {subject}"
        
        # Handle subject details queries
        if query_type == 'subject_details' and subject:
            subject_matches = self.subjects_df[
                self.subjects_df['subject_name'].str.lower().str.contains(subject) | 
                self.subjects_df['subject_abbrevation'].str.lower().str.contains(subject)
            ]
            
            if not subject_matches.empty:
                subject_info = subject_matches.iloc[0]
                return f"Subject details for {subject_info['subject_name']} ({subject_info['subject_abbrevation']}):\n" \
                       f"- Subject Code: {subject_info['subject_code']}\n" \
                       f"- Credits: {subject_info['credits']}\n" \
                       f"- Type: {subject_info['subject_type']}\n" \
                       f"- Branch: {subject_info['branch']}\n" \
                       f"- Semester: {subject_info['semester']}"
        
        # Check if the query is about listing all students
        if query_type == 'student_list' or 'list all students' in question_lower or 'show all students' in question_lower:
            return self._handle_student_list_query(branch)
        
        return None
        
    def _handle_date_specific_attendance(self, date, subject, student_name=None):
        """Handle queries about attendance on a specific date for a subject.
        
        Args:
            date (str): The date to check attendance for
            subject (str): The subject to check attendance for
            student_name (str, optional): If provided, filter for a specific student
            
        Returns:
            str: Formatted response with attendance information
        """
        print(f"Processing date-specific attendance for {date} in {subject}")
        
        # Normalize subject
        normalized_subject = subject.upper() if len(subject) <= 8 else subject.title()
        
        # Get attendance records for this date and subject
        attendance_filter = (
            (self.attendance_df['Date'] == date) &
            ((self.attendance_df['Subject'].str.upper() == normalized_subject) |
             (self.attendance_df['Subject'].str.lower() == subject.lower()))
        )
        
        if student_name:
            # If specific student requested, add that filter
            attendance_filter &= (self.attendance_df['Name'] == student_name)
            
        date_attendance = self.attendance_df[attendance_filter]
        
        if date_attendance.empty:
            if student_name:
                return f"No attendance record found for {student_name} on {date} in {subject.upper()}"
            else:
                return f"No attendance records found for {subject.upper()} on {date}"
        
        # Format the response
        if student_name:
            # Single student
            status = date_attendance.iloc[0]['Status']
            return f"{student_name} was {status} on {date} in {subject.upper()}"
        else:
            # All students
            response = f"Attendance in {subject.upper()} on {date}:\n"
            for _, record in date_attendance.iterrows():
                response += f"- {record['Name']}: {record['Status']}\n"
            return response
        
    def _extract_entities(self, question):
        """Extract entities from the question to identify what's being asked."""
        question = question.lower()  # Ensure question is lowercase for consistent matching
        question_lower = question  # Create question_lower for consistent variable naming
        result = {
            'student_name': None,
            'enrollment_id': None,
            'subject': None,
            'subject_code': None,
            'subject_abbrevation': None,
            'subject_type': None,
            'subject_type_abbreviation': None,
            'query_type': None,
            'date': None,
            'semester': None,  # Added semester entity
            'branch': None,     # Added branch entity
            'credits': None,
            'month_year': None,  # Add month_year for queries about a specific month
        }
        
        possessive_pattern = r'(\w+)\'s'
        possessive_matches = re.findall(possessive_pattern, question)
        if possessive_matches:
            print(f"Found possessive name parts: {possessive_matches}")
            for name_part in possessive_matches:
                # Check if this is a first name of any student
                for full_name in self.students_df['name'].unique():
                    first_name = full_name.split()[0].lower()
                    if name_part.lower() == first_name:
                        result['student_name'] = full_name
                        print(f"Found possessive name reference: {name_part} -> {full_name}")
                        break
        
        # Check for student names directly in the question
        student_names = list(self.students_df['name'].unique())
        for name in student_names:
            if name.lower() in question:
                result['student_name'] = name
                break
        
        # Extract semester information - improved pattern to handle more formats
        sem_pattern = r'(?:sem(?:ester)?|sem\.)\s*(\d+)'
        sem_matches = re.findall(sem_pattern, question)
        if sem_matches:
            result['semester'] = sem_matches[0]
            print(f"Found semester: {result['semester']}")
        else:
            # Try direct numeric reference to semester
            sem_numeric = re.findall(r'(?:in|for|semester)\s+(\d)(?:\b|st|nd|rd|th)', question)
            if sem_numeric:
                result['semester'] = sem_numeric[0]
                print(f"Found numeric semester reference: {result['semester']}")
        
        # Extract branch information
        branch_pattern = r'\b(it|cp|csd)\b'
        branch_matches = re.findall(branch_pattern, question)
        if branch_matches:
            result['branch'] = branch_matches[0].upper()
            print(f"Found branch: {result['branch']}")
        
        # Extract date information first (before enrollment ID) to avoid conflicts
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',                   # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',                   # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',                   # DD-MM-YYYY
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?, \d{4}', # Month Day, Year
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}',  # Month Year
            r'(?:in|during|for) (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}'  # in/during/for Month Year
        ]
        
        # Extract month-year information for broader time range queries
        month_year_patterns = [
            r'(?:in|during|for) (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}',  # in/during/for Month Year
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}'  # Month Year
        ]
        
        # Check for month-year patterns first
        has_month_year_match = False
        for pattern in month_year_patterns:
            month_year_matches = re.findall(pattern, question)
            if month_year_matches:
                # Extract just the month and year
                month_year_text = month_year_matches[0]
                # Remove any leading 'in', 'during', 'for' words
                month_year_text = re.sub(r'^(?:in|during|for)\s+', '', month_year_text)
                result['month_year'] = month_year_text
                has_month_year_match = True
                print(f"Found month-year: {result['month_year']}")
                break
        
        has_date_match = False
        for pattern in date_patterns:
            date_matches = re.findall(pattern, question)
            if date_matches:
                result['date'] = date_matches[0]
                has_date_match = True
                print(f"Found date: {result['date']}")
                break
                
        # Check for years mentioned in the question to avoid treating them as enrollment IDs
        year_patterns = [
            r'\b(20\d{2})\b',                       # Years like 2023, 2024, etc.
            r'\b(19\d{2})\b'                        # Years like 1998, 1999, etc.
        ]
        years_in_question = []
        for pattern in year_patterns:
            year_matches = re.findall(pattern, question)
            years_in_question.extend(year_matches)
        
        # Extract subject - first check for exact quoted subjects
        quoted_subject = None
        if '"' in question:
            # Extract text within double quotes
            quoted_matches = re.findall(r'"([^"]*)"', question)
            if quoted_matches:
                quoted_subject = quoted_matches[0].strip()
        elif "'" in question:
            # Extract text within single quotes
            quoted_matches = re.findall(r"'([^']*)'", question)
            if quoted_matches:
                quoted_subject = quoted_matches[0].strip()
        
        if quoted_subject:
            # Look for exact matches in the quoted text first
            print(f"Found quoted text: '{quoted_subject}'")
            
            # Try to match it against subject names
            subject_matches = self.subjects_df[
                self.subjects_df['subject_name'].str.lower() == quoted_subject.lower()
            ]
            
            if not subject_matches.empty:
                result['subject'] = subject_matches.iloc[0]['subject_name']
                print(f"Found exact subject match from quotes: {result['subject']}")
        
        # If no subject found from quotes, try abbreviations
        if not result['subject']:
            subject_abbrevs = list(self.subjects_df['subject_abbrevation'].unique())
            subject_abbrevs.sort(key=len, reverse=True)  # Sort by length (longest first)
        
        for abbrev in subject_abbrevs:
            if abbrev.lower() in question.split() or f"{abbrev.lower()} " in question or f" {abbrev.lower()}" in question:
                result['subject'] = abbrev
                break
        
        # Check for common abbreviations - especially for subjects
        if not result['subject']:
            common_abbrevs = ['ml', 'ai', 'os', 'iot', 'coa', 'dbms', 'daa', 'ins', 'scm', 'awd', 'lavcode', 'pwp']
            for abbrev in common_abbrevs:
                if abbrev.lower() in question.split() or f"{abbrev.lower()} " in question or f" {abbrev.lower()}" in question:
                    result['subject'] = abbrev
                    break
        
        # If no subject found from quotes or abbreviations, check for subject names
        if not result['subject']:
            subject_names = list(self.subjects_df['subject_name'].unique())
            # Sort by length (longest first) to prefer more specific matches
            subject_names.sort(key=len, reverse=True)
            for name in subject_names:
                if name.lower() in question:
                    result['subject'] = name
                    print(f"Found subject name: {name}")
                break
        
        # Check for enrollment IDs - AFTER checking for dates to avoid confusion
        enrollment_pattern = r'(\d{14})'  # Pattern for 14-digit enrollment numbers
        enrollment_matches = re.findall(enrollment_pattern, question)
        if enrollment_matches:
            result['enrollment_id'] = enrollment_matches[0]
        elif any(word in question for word in ['enrollment', 'id']) and not has_date_match:
            # Only look for enrollment numbers if explicitly mentioned and no date was found
            # Try to find any digit sequence that could be an ID, avoiding years that might be in dates
            numbers = re.findall(r'\b\d+\b', question)
            if numbers:
                potential_ids = [num for num in numbers if num not in years_in_question]
                if potential_ids:
                    result['enrollment_id'] = max(potential_ids, key=len)  # Use the longest number sequence
        
        # Identify query type
        if any(phrase in question for phrase in ['enrollment', 'enrollment number', 'enrollment id', 'id of']):
            result['query_type'] = 'enrollment'
        elif any(phrase in question for phrase in ['contact', 'contact info', 'details of student', 'contact information']):
            result['query_type'] = 'contact'
        elif 'email' in question or 'e-mail' in question:
            result['query_type'] = 'email'
        elif any(word in question for word in ['phone', 'number', 'mobile']):
            result['query_type'] = 'phone'
        elif any(phrase in question for phrase in ['subject code', 'course code', 'code for', 'code of']):
            result['query_type'] = 'subject_code'
        elif any(phrase in question for phrase in ['subject details', 'tell me about subject', 'information about']):
            result['query_type'] = 'subject_details'
        elif "total classes" in question_lower and result['subject']:
            result['query_type'] = 'class_count'
        elif "zero credit" in question_lower or "zero-credit" in question_lower:
            result['query_type'] = 'zero_credit'
        elif "compare attendance" in question_lower:
            students = [s.strip() for s in question.split("compare attendance of")[1].split("and")]
            return self._handle_comparison_query(students[0], students[1], result['subject'])
        elif "missed attendance on" in question_lower:
            date = result['date']
            return self._handle_date_absence_query(date)
        elif "total credits" in question_lower and result['semester']:
            return self._handle_semester_credits_query(result['semester'])
        
        # Check for attendance-related queries
        if any(word in question for word in ['attendance', 'present', 'absent', 'percentage', 'attend']):
            result['query_type'] = 'attendance'
            
            # Check for lowest/highest attendance queries
            if any(word in question for word in ['lowest', 'minimum', 'worst']):
                result['query_type'] = 'lowest_attendance'
            elif any(word in question for word in ['highest', 'maximum', 'best']):
                result['query_type'] = 'highest_attendance'
                
            # Check if we're asking for a list of attendance
            if 'list' in question or 'all subjects' in question:
                result['query_type'] = 'attendance_list'
        
        # Check for elective course queries
        if 'elective' in question or 'electives' in question:
            result['query_type'] = 'electives'
            # Determine if we're asking for professional or open electives
            if 'professional' in question or 'pe' in question:
                result['subject'] = 'Professional Elective'
            elif 'open' in question or 'oe' in question:
                result['subject'] = 'Open Elective'
        
        # Check for full form queries
        if 'full form' in question:
            # This could be asking for full form of an abbreviation
            result['query_type'] = 'full_form'
            
            # Try to identify the abbreviation being asked about
            words = question.split()
            for word in words:
                if word.upper() == word and len(word) >= 2 and word.isalpha():
                    result['subject'] = word.lower()
                    break
            
            # Also check for common subject abbreviations
            common_abbrevs = ['ml', 'ai', 'os', 'iot', 'coa', 'dbms', 'daa']
            for abbrev in common_abbrevs:
                if abbrev in question:
                    result['subject'] = abbrev
                    break
                    
        # Check for credit-related queries
        if any(word in question for word in ['credit', 'credits']):
            result['query_type'] = 'credits'
            
        # Check for class count queries
        if any(phrase in question for phrase in ['total classes', 'how many classes', 'number of classes']):
            result['query_type'] = 'class_count'
            
        # Check for subject list by semester queries
        if (('subjects' in question or 'courses' in question) and 
            ('offered' in question or 'available' in question or 'in semester' in question or 'in sem' in question)):
            result['query_type'] = 'semester_subjects'
            
        # Check for student list queries
        if 'list all students' in question or 'show all students' in question:
            result['query_type'] = 'student_list'
            
        # Check for course structure queries (credit distribution, etc.)
        if 'credit distribution' in question or 'course structure' in question:
            result['query_type'] = 'course_structure'
            
        # Check for analysis types of queries
        if ('total number' in question or 'how many' in question) and ('subjects' in question or 'courses' in question):
            if result['semester'] is not None:
                result['query_type'] = 'semester_subject_count'
            else:
                result['query_type'] = 'subject_count'
        
        if "compare" in question_lower:
            result['query_type'] = 'comparison'
                
        return result
    
    def _find_best_name_match(self, query, names):
        """Find the best matching name using fuzzy matching."""
        best_match = None
        highest_score = 0
        query = query.lower()  # Ensure query is lowercase for consistent matching
        
        for name in names:
            score = fuzz.ratio(query, name.lower())
            if score > highest_score and score > 70:  # Threshold adjusted
                highest_score = score
                best_match = name
        
        # Also check for first name matches
        if not best_match:
            for name in names:
                first_name = name.split()[0].lower()
                score = fuzz.ratio(query, first_name)
                if score > highest_score and score > 70:
                    highest_score = score
                    best_match = name
        
        return best_match
    
    def get_answer(self, question):
        """Find answer using RAG pipeline with fallback to similarity search."""
        print(f"\n=== Processing query: {question} ===")
        
        # First try to get an answer using real-time data analysis
        real_time_answer = self.get_real_time_answer(question)
        if real_time_answer:
            print(f"Using real-time answer: {real_time_answer}")
            return real_time_answer
            
        # Then try RAG pipeline
        try:
            print("Trying RAG pipeline...")
            rag_answer = self.qa_chain.invoke(question)['result']
            if rag_answer and not any(phrase in rag_answer.lower() for phrase in 
                                      ["i don't know", "i don't have enough information", 
                                       "i don't have access", "i'm not able to"]):
                print(f"Using RAG answer: {rag_answer}")
                return rag_answer
            else:
                print("RAG gave an 'I don't know' answer")
        except Exception as e:
            print(f"RAG error: {e}")
        
        # As a last resort, use the legacy model-based approach
        if self.legacy_model:
            print("Trying legacy model...")
            question_vector = self.legacy_model['vectorizer'].transform([question])
            similarities = cosine_similarity(question_vector, self.legacy_model['question_vectors']).flatten()
            max_sim_idx = np.argmax(similarities)
            max_sim = similarities[max_sim_idx]
            
            if max_sim >= 0.5:
                answer = self.legacy_model['answers'][max_sim_idx]
                print(f"Using legacy answer with similarity {max_sim:.2f}: {answer}")
                return answer
            # Update get_answer() to include new question patterns in the legacy model
            elif "Professional Electives across all semesters" in question:
                # Count professional electives from the subjects dataframe
                pe_count = len(self.subjects_df[self.subjects_df['subject_type'] == 'Professional Elective'])
                return f"There are {pe_count} Professional Elective subjects available"
            else:
                print(f"Legacy model similarity too low: {max_sim:.2f}")
        
        # If all methods fail, return a fallback response
        return "I'm not sure how to answer that. Could you rephrase your question?"
    
    def _calculate_subject_attendance(self, subject):
        """Calculate attendance percentage for all students in a specific subject."""
        # Normalize subject to handle both abbreviations and full names
        normalized_subject = subject.upper() if len(subject) <= 8 else subject.title()
        
        # Find attendance records for this subject
        subject_attendance = self.attendance_df[
            (self.attendance_df['Subject'].str.upper() == normalized_subject) |
            (self.attendance_df['Subject'].str.lower() == subject.lower())
        ]
        
        if subject_attendance.empty:
            # Try partial matching if no exact match
            for subj in self.attendance_df['Subject'].unique():
                if subject.lower() in subj.lower():
                    subject_attendance = self.attendance_df[self.attendance_df['Subject'] == subj]
                    break
        
        if subject_attendance.empty:
            return None
        
        # Calculate attendance percentage for each student
        student_percentages = []
        for student_name in subject_attendance['Name'].unique():
            student_records = subject_attendance[subject_attendance['Name'] == student_name]
            attendance_pct = (student_records['Status'] == 'Present').mean() * 100
            student_percentages.append({'Name': student_name, 'Status': attendance_pct})
        
        return pd.DataFrame(student_percentages)
    
    def _handle_elective_queries(self, semester, elective_type=None):
        """Handle queries about elective courses for a specific semester.
        
        Args:
            semester (str): The semester number to query
            elective_type (str, optional): Type of elective ('Professional Elective' or 'Open Elective')
                            
        Returns:
            str: Formatted response with elective courses
        """
        print(f"Processing electives query for semester {semester}")
        
        # Make sure semester is a string
        semester = str(semester)
        
        # Filter subjects by semester
        sem_subjects = self.subjects_df[self.subjects_df['semester'] == semester]
        
        if elective_type:
            # Further filter by elective type (Professional or Open)
            electives = sem_subjects[sem_subjects['subject_type'] == elective_type]
        else:
            # Get all electives (both Professional and Open)
            electives = sem_subjects[sem_subjects['subject_type'].str.contains('Elective')]
        
        if not electives.empty:
            response = f"Elective courses for semester {semester}:\n"
            for _, elective in electives.iterrows():
                response += f"- {elective['subject_name']} ({elective['subject_abbrevation']}): {elective['subject_type']}\n"
            return response
        return f"No elective courses found for semester {semester}"
    
    def _handle_student_attendance_list(self, student_name):
        """Generate a list of a student's attendance percentages across all subjects.
        
        Args:
            student_name (str): The student's name
            
        Returns:
            str: Formatted response with attendance percentages by subject
        """
        print(f"Processing attendance list for {student_name}")
        response = f"{student_name}'s attendance percentages by subject:\n"
        
        # Get all subjects this student has attendance records for
        student_attendance = self.attendance_df[self.attendance_df['Name'] == student_name]
        if student_attendance.empty:
            return f"No attendance records found for {student_name}"
        
        # Calculate attendance percentage for each subject
        subject_percentages = {}
        for subject_name in student_attendance['Subject'].unique():
            subject_records = student_attendance[student_attendance['Subject'] == subject_name]
            percentage = (subject_records['Status'] == 'Present').mean() * 100
            subject_percentages[subject_name] = percentage
        
        # Sort by percentage (descending)
        sorted_subjects = sorted(subject_percentages.items(), key=lambda x: x[1], reverse=True)
        
        # Build the response
        for subject_name, percentage in sorted_subjects:
            response += f"- {subject_name}: {percentage:.1f}%\n"
            
        return response
    
    def _handle_extreme_attendance_query(self, student_name, query_type):
        """Find the subject with highest or lowest attendance for a student.
        
        Args:
            student_name (str): The student's name
            query_type (str): Either 'lowest_attendance' or 'highest_attendance'
            
        Returns:
            str: Formatted response with the extreme attendance subject
        """
        print(f"Processing {query_type} for {student_name}")
        
        # Get all subjects this student has attendance records for
        student_attendance = self.attendance_df[self.attendance_df['Name'] == student_name]
        if student_attendance.empty:
            return f"No attendance records found for {student_name}"
        
        # Calculate attendance percentage for each subject
        subject_percentages = {}
        for subject_name in student_attendance['Subject'].unique():
            subject_records = student_attendance[student_attendance['Subject'] == subject_name]
            percentage = (subject_records['Status'] == 'Present').mean() * 100
            subject_percentages[subject_name] = percentage
        
        # Find the subject with lowest/highest attendance
        if query_type == 'lowest_attendance':
            subject_name = min(subject_percentages.items(), key=lambda x: x[1])[0]
            percentage = subject_percentages[subject_name]
            return f"{student_name} has the lowest attendance in {subject_name}: {percentage:.1f}%"
        else:  # highest_attendance
            subject_name = max(subject_percentages.items(), key=lambda x: x[1])[0]
            percentage = subject_percentages[subject_name]
            return f"{student_name} has the highest attendance in {subject_name}: {percentage:.1f}%"
    
    def _handle_subject_list_query(self, semester=None, branch=None):
        """Handle queries about subjects offered in a specific semester and branch.
        
        Args:
            semester (str): Semester number
            branch (str): Branch code (IT, CP, CSD)
            
        Returns:
            str: Formatted response with subject information
        """
        print(f"Processing subject list query for semester {semester}, branch {branch}")
        
        # Convert semester to string if it's not None
        if semester is not None:
            semester = str(semester)
        
        # Filter the subjects dataframe
        filtered_subjects = self.subjects_df
        
        if semester is not None:
            filtered_subjects = filtered_subjects[filtered_subjects['semester'] == semester]
            
        if branch is not None:
            filtered_subjects = filtered_subjects[filtered_subjects['branch'].str.lower() == branch.lower()]
            
        if filtered_subjects.empty:
            if semester and branch:
                return f"No subjects found for {branch.upper()} in Semester {semester}"
            elif semester:
                return f"No subjects found for Semester {semester}"
            elif branch:
                return f"No subjects found for {branch.upper()}"
            else:
                return "No subjects found with the given criteria"
                
        # Group subjects by type
        regular_subjects = filtered_subjects[filtered_subjects['subject_type'] == 'Regular']
        pe_subjects = filtered_subjects[filtered_subjects['subject_type'] == 'Professional Elective']
        oe_subjects = filtered_subjects[filtered_subjects['subject_type'] == 'Open Elective']
        
        # Format response with explicit mention of semester and branch in header
        if semester and branch:
            response = f"Subjects for {branch.upper()} Semester {semester}:\n\n"
        elif semester:
            response = f"Subjects for Semester {semester}:\n\n"
        elif branch:
            response = f"Subjects for {branch.upper()}:\n\n"
        else:
            response = "All Subjects:\n\n"
            
        # Create a simple list of all subjects first
        subject_list = ", ".join(filtered_subjects['subject_name'].tolist())
        response += f"All subjects: {subject_list}\n\n"
            
        if not regular_subjects.empty:
            response += "Regular Subjects:\n"
            for _, subject in regular_subjects.iterrows():
                response += f"- {subject['subject_name']} ({subject['subject_abbrevation']}): {subject['credits']} credits\n"
            response += "\n"
            
        if not pe_subjects.empty:
            response += "Professional Electives:\n"
            for _, subject in pe_subjects.iterrows():
                response += f"- {subject['subject_name']} ({subject['subject_abbrevation']}): {subject['credits']} credits\n"
            response += "\n"
            
        if not oe_subjects.empty:
            response += "Open Electives:\n"
            for _, subject in oe_subjects.iterrows():
                response += f"- {subject['subject_name']} ({subject['subject_abbrevation']}): {subject['credits']} credits\n"
                
        return response.strip()
        
    def _handle_student_list_query(self, branch=None):
        """Handle queries to list all students, optionally filtered by branch.
        
        Args:
            branch (str): Branch to filter by
            
        Returns:
            str: Formatted list of students
        """
        print(f"Processing student list query for branch {branch}")
        
        if branch:
            filtered_students = self.students_df[self.students_df['course'].str.lower() == branch.lower()]
        else:
            filtered_students = self.students_df
            
        if filtered_students.empty:
            return f"No students found {f'in {branch.upper()}' if branch else ''}"
            
        response = f"Students {f'in {branch.upper()}' if branch else ''}:\n"
        for _, student in filtered_students.iterrows():
            response += f"- {student['name']} (ID: {student['student_id']})\n"
            
        return response
        
    def _handle_credit_query(self, subject=None, semester=None):
        """Handle queries about subject credits or credit distribution.
        
        Args:
            subject (str): Subject name or abbreviation
            semester (str): Semester number
            
        Returns:
            str: Formatted response with credit information
        """
        print(f"Processing credit query for subject {subject}, semester {semester}")
        
        # If asking about a specific subject
        if subject:
            subject_info = None
            
            # Try to find by abbreviation first
            matches = self.subjects_df[self.subjects_df['subject_abbrevation'].str.lower() == subject.lower()]
            if not matches.empty:
                subject_info = matches.iloc[0]
            else:
                # Try to find by partial name match
                matches = self.subjects_df[self.subjects_df['subject_name'].str.lower().str.contains(subject.lower())]
                if not matches.empty:
                    subject_info = matches.iloc[0]
                    
            if subject_info is not None:
                return f"{subject_info['subject_name']} ({subject_info['subject_abbrevation']}) carries {subject_info['credits']} credits"
            else:
                return f"Subject '{subject}' not found"
                
        # If asking about a semester
        if semester:
            sem_subjects = self.subjects_df[self.subjects_df['semester'] == semester]
            if sem_subjects.empty:
                return f"No subjects found for Semester {semester}"
                
            total_credits = sem_subjects['credits'].astype(int).sum()
            
            # Create breakdown by subject
            response = f"Credit distribution for Semester {semester}:\n"
            response += f"Total credits: {total_credits}\n\n"
            response += "Breakdown by subject:\n"
            
            for _, subj in sem_subjects.iterrows():
                response += f"- {subj['subject_name']} ({subj['subject_abbrevation']}): {subj['credits']} credits\n"
                
            return response
            
        return "Please specify a subject or semester for credit information"
        
    def _handle_class_count_query(self, subject):
        """Handle queries about total classes for a subject.
        
        Args:
            subject (str): Subject name or abbreviation
            
        Returns:
            str: Number of total classes for the subject
        """
        print(f"Processing class count query for {subject}")
        
        if not subject:
            return "Please specify a subject to get the class count"
            
        # Check if subject might be a quoted string and clean it
        if isinstance(subject, str):
            if subject.startswith('"') and subject.endswith('"'):
                subject = subject[1:-1].strip()
            elif subject.startswith("'") and subject.endswith("'"):
                subject = subject[1:-1].strip()
        
        try:
            # Try by exact abbreviation match first
            matches = self.subjects_df[
                self.subjects_df['subject_abbrevation'].str.lower() == subject.lower()
            ]
            
            # If no match, try exact name match
            if matches.empty:
                matches = self.subjects_df[
                    self.subjects_df['subject_name'].str.lower() == subject.lower()
                ]
            
            # If still no match, try partial name match
            if matches.empty:
                matches = self.subjects_df[
                    self.subjects_df['subject_name'].str.lower().str.contains(subject.lower())
                ]
            
            # Process the results
            if not matches.empty:
                subject_info = matches.iloc[0]
                subject_name = subject_info['subject_name']
                subject_abbrev = subject_info['subject_abbrevation']
                total_classes = subject_info['total_classes']
                return f"{subject_name} ({subject_abbrev}) has {total_classes} total classes"
            else:
                return f"Subject '{subject}' not found"
        
        except Exception as e:
            print(f"Error in class count query: {e}")
            # Fall back to a safer method
            try:
                for _, row in self.subjects_df.iterrows():
                    if (row['subject_name'].lower() == subject.lower() or 
                        row['subject_abbrevation'].lower() == subject.lower()):
                        return f"{row['subject_name']} ({row['subject_abbrevation']}) has {row['total_classes']} total classes"
                return f"Subject '{subject}' not found or error accessing class count"
            except:
                return f"Could not retrieve class count information for '{subject}'"
            
    def _handle_course_structure_query(self, semester):
        """Handle queries about course structure and credit distribution.
        
        Args:
            semester (str): Semester number
            
        Returns:
            str: Formatted response with course structure
        """
        print(f"Processing course structure query for semester {semester}")
        
        if not semester:
            return "Please specify a semester to get the course structure"
            
        sem_subjects = self.subjects_df[self.subjects_df['semester'] == semester]
        if sem_subjects.empty:
            return f"No subjects found for Semester {semester}"
            
        # Calculate total credits
        total_credits = sem_subjects['credits'].astype(int).sum()
        
        # Count subjects by type
        regular_count = len(sem_subjects[sem_subjects['subject_type'] == 'Regular'])
        pe_count = len(sem_subjects[sem_subjects['subject_type'] == 'Professional Elective'])
        oe_count = len(sem_subjects[sem_subjects['subject_type'] == 'Open Elective'])
        
        # Format response
        response = f"Course Structure for Semester {semester}:\n"
        response += f"Total Credits: {total_credits}\n"
        response += f"Regular Subjects: {regular_count}\n"
        response += f"Professional Elective Subjects: {pe_count}\n"
        response += f"Open Elective Subjects: {oe_count}\n\n"
        
        response += "Subject Details:\n"
        for _, subj in sem_subjects.iterrows():
            response += f"- {subj['subject_name']} ({subj['subject_abbrevation']}): {subj['credits']} credits, Type: {subj['subject_type']}\n"
            
        return response
            
    def _handle_subject_count_query(self, semester=None, subject_type=None):
        """Handle queries about counts of different types of subjects.
        
        Args:
            semester (str): Optional semester number to filter by
            subject_type (str): Optional subject type to count ('Regular', 'Professional Elective', 'Open Elective')
            
        Returns:
            str: Count information
        """
        print(f"Processing subject count query for semester {semester}, type {subject_type}")
        
        filtered_subjects = self.subjects_df
        
        if semester:
            filtered_subjects = filtered_subjects[filtered_subjects['semester'] == semester]
            
        if subject_type:
            filtered_subjects = filtered_subjects[filtered_subjects['subject_type'] == subject_type]
            
        count = len(filtered_subjects)
        
        if semester and subject_type:
            return f"There are {count} {subject_type} subjects in Semester {semester}"
        elif semester:
            return f"There are {count} total subjects in Semester {semester}"
        elif subject_type:
            return f"There are {count} {subject_type} subjects across all semesters"
        else:
            return f"There are {count} total subjects in the database"
        
    def _handle_date_absence_query(self, date):
        """Find students absent across all subjects on a specific date.
        
        Args:
            date (str): The date to check for absences
            
        Returns:
            str: List of students absent on that date
        """
        if not date:
            return "Please specify a date to check for absences"
            
        print(f"Processing absence query for date: {date}")
        
        try:
            absent_students = []
            for student in self.students_df['name'].unique():
                attendance = self.attendance_df[
                (self.attendance_df['Name'] == student) &
                (self.attendance_df['Date'] == date) &
                (self.attendance_df['Status'] == 'Absent')
                ]
                if not attendance.empty:
                    absent_students.append(student)
                    
            if absent_students:
                return f"Students absent on {date}: {', '.join(absent_students)}"
            return f"No absences recorded on {date}"
            
        except Exception as e:
            print(f"Error in date absence query: {e}")
            return f"Error processing absence query for {date}: {str(e)}"

    def _handle_comparison_query(self, student1, student2, subject):
        """Compare attendance between two students in a subject"""
        pct1 = self._get_student_subject_attendance(student1, subject)
        pct2 = self._get_student_subject_attendance(student2, subject)
    
        if pct1 and pct2:
            return f"Attendance comparison in {subject}:\n- {student1}: {pct1}%\n- {student2}: {pct2}%"
        return "Comparison data not available"

    def _handle_zero_credit_query(self, semester):
        """Handle queries about zero-credit subjects for a specific semester.
        
        Args:
            semester (str or int): Semester number
            
        Returns:
            str: Information about zero-credit subjects in the specified semester
        """
        print(f"Processing zero credit query for semester {semester}")
        
        try:
            # Convert semester to int
            if semester is not None:
                if isinstance(semester, str):
                    # Clean any extra text, extract just the number
                    sem_match = re.search(r'\d+', semester)
                    if sem_match:
                        semester = int(sem_match.group())
                    else:
                        return f"Couldn't understand semester format: {semester}"
                elif isinstance(semester, int):
                    semester = int(semester)
                else:
                    return f"Invalid semester format: {semester}"
            else:
                return "Please specify a semester to check for zero-credit subjects"
            
            # Filter subjects for the specified semester and zero credits
            zero_credit_subjects = self.subjects_df[
                (self.subjects_df['semester'] == semester) & 
                (self.subjects_df['credits'] == '0')
            ]
            
            if zero_credit_subjects.empty:
                return f"There are no zero-credit subjects in Semester {semester}."
            else:
                zero_credit_list = []
                for _, subject in zero_credit_subjects.iterrows():
                    zero_credit_list.append(f"{subject['subject_name']} ({subject['subject_abbrevation']})")
                
                if len(zero_credit_list) == 1:
                    return f"There is 1 zero-credit subject in Semester {semester}: {zero_credit_list[0]}"
                else:
                    subjects_str = ", ".join(zero_credit_list)
                    return f"There are {len(zero_credit_list)} zero-credit subjects in Semester {semester}: {subjects_str}"
        except Exception as e:
            print(f"Error in zero credit query: {e}")
            return f"Error processing zero-credit query for Semester {semester}: {str(e)}"

    def _handle_semester_credits_query(self, semester):
        """Calculate total credits for a semester and show distribution."""
        # Convert semester to string if it's not already
        semester = str(semester)
        
        # Get subjects for the semester
        sem_subjects = self.subjects_df[self.subjects_df['semester'] == semester]
        
        if sem_subjects.empty:
            return f"No subjects found for Semester {semester}"
        
        # Calculate total credits
        total = sem_subjects['credits'].astype(int).sum()
        
        # Create detailed response with distribution
        response = f"Credit distribution for Semester {semester}:\n"
        response += f"Total credits: {total}\n\n"
        response += "Breakdown by subject:\n"
        
        # Sort subjects alphabetically
        sem_subjects = sem_subjects.sort_values(by='subject_name')
        
        for _, subject in sem_subjects.iterrows():
            response += f"- {subject['subject_name']} ({subject['subject_abbrevation']}): {subject['credits']} credits\n"
            
        return response
        
    def _get_student_subject_attendance(self, student_name, subject):
        """Calculate a student's attendance percentage in a specific subject.
        
        Args:
            student_name (str): The student's name
            subject (str): The subject name or abbreviation
            
        Returns:
            float or None: The attendance percentage or None if no records found
        """
        # Normalize subject
        normalized_subject = subject.upper() if len(subject) <= 8 else subject.title()
        
        # Get student's attendance in this subject
        attendance = self.attendance_df[
            (self.attendance_df['Name'] == student_name) &
            ((self.attendance_df['Subject'].str.upper() == normalized_subject) |
             (self.attendance_df['Subject'].str.lower() == subject.lower()))
        ]
        
        if attendance.empty:
            return None
        
        # Calculate percentage
        percentage = (attendance['Status'] == 'Present').mean() * 100
        return round(percentage, 1)
    
    def _handle_month_year_attendance(self, month_year, subject, student_name=None):
        """Handle queries about attendance over a month and year.
        
        Args:
            month_year (str): The month and year, e.g., "April 2025"
            subject (str): The subject to check attendance for
            student_name (str, optional): If provided, filter for a specific student
            
        Returns:
            str: Formatted response with attendance information
        """
        print(f"Processing month-year attendance for {month_year} in {subject}")
        
        # Parse the month and year
        month_year_lower = month_year.lower()
        
        # Extract month name
        month_pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*'
        month_match = re.search(month_pattern, month_year_lower)
        if not month_match:
            return f"Could not understand the month in '{month_year}'"
            
        month_abbr = month_match.group(1)
        month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        month_num = month_map.get(month_abbr)
        
        # Extract year
        year_match = re.search(r'(\d{4})', month_year)
        if not year_match:
            return f"Could not understand the year in '{month_year}'"
            
        year = year_match.group(1)
        
        # Normalize subject
        normalized_subject = subject.upper() if len(subject) <= 8 else subject.title()
        
        # Get attendance records for this month-year and subject
        attendance_filter = (
            (self.attendance_df['Date'].str.contains(f'{year}-{month_num}')) &
            ((self.attendance_df['Subject'].str.upper() == normalized_subject) |
             (self.attendance_df['Subject'].str.lower() == subject.lower()))
        )
        
        if student_name:
            # If specific student requested, add that filter
            attendance_filter &= (self.attendance_df['Name'] == student_name)
            
        month_attendance = self.attendance_df[attendance_filter]
        
        if month_attendance.empty:
            if student_name:
                return f"No attendance records found for {student_name} in {subject.upper()} during {month_year}"
            else:
                return f"No attendance records found for {subject.upper()} during {month_year}"
        
        # Format the response
        if student_name:
            # Single student - check if they attended all classes
            total_classes = len(month_attendance)
            attended_classes = len(month_attendance[month_attendance['Status'] == 'Present'])
            
            if attended_classes == total_classes:
                return f"Yes, {student_name} attended all {total_classes} {subject.upper()} classes in {month_year}"
            elif attended_classes == 0:
                return f"No, {student_name} missed all {total_classes} {subject.upper()} classes in {month_year}"
            else:
                return f"{student_name} attended {attended_classes} out of {total_classes} {subject.upper()} classes in {month_year}"
        else:
            # All students summary
            response = f"Attendance in {subject.upper()} during {month_year}:\n"
            
            # Group by student and count attendance
            student_summary = {}
            for student in month_attendance['Name'].unique():
                student_records = month_attendance[month_attendance['Name'] == student]
                total = len(student_records)
                present = len(student_records[student_records['Status'] == 'Present'])
                student_summary[student] = (present, total)
            
            # Format results
            for student, (present, total) in student_summary.items():
                percentage = (present / total) * 100 if total > 0 else 0
                response += f"- {student}: {present}/{total} classes ({percentage:.1f}%)\n"
                
            return response
            
    def _handle_student_absence_query(self, student_name, subject):
        """Handle queries about which dates a student missed classes in a subject.
        
        Args:
            student_name (str): The student's name
            subject (str): The subject to check
            
        Returns:
            str: Formatted response with absence information
        """
        print(f"Processing absence query for {student_name} in {subject}")
        
        # Normalize subject
        normalized_subject = subject.upper() if len(subject) <= 8 else subject.title()
        
        # Get all attendance records for this student and subject
        attendance_filter = (
            (self.attendance_df['Name'] == student_name) &
            ((self.attendance_df['Subject'].str.upper() == normalized_subject) |
             (self.attendance_df['Subject'].str.lower() == subject.lower()))
        )
            
        student_attendance = self.attendance_df[attendance_filter]
        
        if student_attendance.empty:
            return f"No attendance records found for {student_name} in {subject.upper()}"
            
        # Get dates when student was absent
        absent_dates = student_attendance[student_attendance['Status'] == 'Absent']['Date'].tolist()
        
        if not absent_dates:
            total_classes = len(student_attendance)
            return f"{student_name} has not missed any {subject.upper()} classes. Attended all {total_classes} classes."
        else:
            # Sort the dates
            absent_dates.sort()
            
            if len(absent_dates) == 1:
                return f"{student_name} missed {subject.upper()} class on: {absent_dates[0]}"
            else:
                dates_str = ", ".join(absent_dates)
                return f"{student_name} missed {subject.upper()} classes on these dates: {dates_str}"
    
    def _handle_subject_type_query(self, subject):
        """Handle queries about what type of subject something is (elective, regular, etc).
        
        Args:
            subject (str): The subject name or abbreviation
            
        Returns:
            str: Information about the subject type
        """
        print(f"Checking subject type for '{subject}'")
        
        # First try direct string containment to handle partial names better
        subject_matches = self.subjects_df[
            self.subjects_df['subject_name'].str.lower().str.contains(subject.lower())
        ]
        
        # If that doesn't work, try exact match on name
        if subject_matches.empty:
            subject_matches = self.subjects_df[
                self.subjects_df['subject_name'].str.lower() == subject.lower()
            ]
        
        # If still no match, try by abbreviation
        if subject_matches.empty:
            subject_matches = self.subjects_df[
                self.subjects_df['subject_abbrevation'].str.lower() == subject.lower()
            ]
        
        # Add debug prints
        print(f"Subject matches found: {len(subject_matches)}")
        if not subject_matches.empty:
            for i in range(min(3, len(subject_matches))):
                print(f"Match {i+1}: {subject_matches.iloc[i]['subject_name']} (Type: {subject_matches.iloc[i]['subject_type']})")
        
        if not subject_matches.empty:
            subject_info = subject_matches.iloc[0]
            subject_type = subject_info['subject_type']
            
            if subject_type == 'Regular':
                return f"'{subject_info['subject_name']}' is a Regular subject, not an elective."
            elif subject_type == 'Professional Elective':
                return f"'{subject_info['subject_name']}' is a Professional Elective."
            elif subject_type == 'Open Elective':
                return f"'{subject_info['subject_name']}' is an Open Elective."
            else:
                return f"'{subject_info['subject_name']}' is classified as: {subject_type}"
        else:
            return f"Subject '{subject}' not found in the database."
    
    def run_gui(self):
        """Run the chatbot GUI."""
        # Create the main window
        root = tk.Tk()
        root.title("Attendance Chatbot")
        root.geometry("800x600")
        root.config(bg="#f0f0f0")
        
        # Create and configure the main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a style for the UI elements
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TButton", background="#4CAF50", foreground="black", font=("Arial", 12))
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 12))
        
        # Add a title label
        title_label = ttk.Label(main_frame, text="Attendance Management Chatbot", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create a frame for the chat display
        chat_frame = ttk.Frame(main_frame)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create the chat display area
        chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED, 
                                               font=("Arial", 12))
        chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for user input
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create the input text field
        input_text = ttk.Entry(input_frame, font=("Arial", 12))
        input_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Add instruction text to the chat display
        def display_welcome():
            chat_display.config(state=tk.NORMAL)
            chat_display.insert(tk.END, "Welcome to the Attendance Management Chatbot!\n\n")
            chat_display.insert(tk.END, "You can ask questions about:\n")
            chat_display.insert(tk.END, "- Student attendance (overall or by subject)\n")
            chat_display.insert(tk.END, "- Subject codes and details\n")
            chat_display.insert(tk.END, "- Student information (enrollment, contact details)\n\n")
            chat_display.insert(tk.END, "For example, try asking:\n")
            chat_display.insert(tk.END, "- What is (studnet name)'s attendance in ML?\n")
            chat_display.insert(tk.END, "- What is the subject code for Machine Learning?\n")
            chat_display.insert(tk.END, "- Show me attendance percentages in DBMS\n\n")
            chat_display.insert(tk.END, "Chatbot: How can I help you today?\n\n")
            chat_display.config(state=tk.DISABLED)
            chat_display.see(tk.END)
        
        # Define the function to handle user input
        def process_input(event=None):
            user_question = input_text.get()
            if not user_question.strip():
                return
            
            # Display user question
            chat_display.config(state=tk.NORMAL)
            chat_display.insert(tk.END, f"You: {user_question}\n\n")
            chat_display.config(state=tk.DISABLED)
            chat_display.see(tk.END)
            
            # Clear the input field
            input_text.delete(0, tk.END)
            
            # Get the answer and handle potential errors
            try:
                answer = self.get_answer(user_question)
                
                # Display the chatbot's response
                chat_display.config(state=tk.NORMAL)
                chat_display.insert(tk.END, f"Chatbot: {answer}\n\n")
                chat_display.config(state=tk.DISABLED)
                chat_display.see(tk.END)
            except Exception as e:
                # Handle any errors gracefully
                chat_display.config(state=tk.NORMAL)
                chat_display.insert(tk.END, f"Chatbot: Sorry, I encountered an error: {str(e)}\n\n")
                chat_display.config(state=tk.DISABLED)
                chat_display.see(tk.END)
                print(f"Error processing input: {e}")
        
        # Create the submit button
        submit_button = ttk.Button(input_frame, text="Send", command=process_input)
        submit_button.pack(side=tk.RIGHT)
        
        # Bind the Enter key to the process_input function
        input_text.bind("<Return>", process_input)
        
        # Display the welcome message
        display_welcome()
        
        # Set focus to the input field
        input_text.focus_set()
        
        # Start the GUI event loop
        root.mainloop()

    def _handle_students_missing_on_date(self, date):
        """Find all students who were absent on a specific date, across all subjects.
        
        Args:
            date (str): The date to check (YYYY-MM-DD format)
            
        Returns:
            str: A list of students who were absent on that date
        """
        if not date:
            return "Please provide a date to check for absences"
            
        print(f"Checking for absences on {date}")
        
        try:
            # Find all students marked as absent on this date
            absent_records = self.attendance_df[(self.attendance_df['Date'] == date) & 
                                               (self.attendance_df['Status'] == 'Absent')]
            
            if absent_records.empty:
                return f"No students were absent on {date}"
                
            # Get unique list of students who were absent
            absent_students = absent_records['Name'].unique().tolist()
            
            if len(absent_students) == 1:
                return f"1 student was absent on {date}: {absent_students[0]}"
            else:
                return f"{len(absent_students)} students were absent on {date}: {', '.join(absent_students)}"
                
        except Exception as e:
            print(f"Error checking for absences: {e}")
            return f"Error processing absence query for {date}: {str(e)}"


chatbot_instance = AttendanceChatbot()

def get_chat_response(question):
    """Web interface for getting chatbot responses"""
    try:
        # Refresh data before processing question
        chatbot_instance.refresh_data()
        return chatbot_instance.get_answer(question)
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Create and run the chatbot application if this script is run directly
if __name__ == "__main__":
    # Create the chatbot
    chatbot = AttendanceChatbot()
    
    # Run the GUI
    chatbot.run_gui()