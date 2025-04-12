import pickle
import os
import tkinter as tk
from tkinter import scrolledtext, ttk
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import openpyxl
import subprocess
import re
from train_chatbot import load_attendance_data, load_student_data, load_subject_data, calculate_attendance_stats

class AttendanceChatbot:
    def __init__(self):
        self.model = self.load_model()
        self.refresh_data()
        
    def load_model(self):
        """Load the trained chatbot model."""
        try:
            with open('models/chatbot_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            print("Model file not found. Please run train_chatbot.py first.")
            return None
    
    def refresh_data(self):
        """Load fresh data from all sources."""
        self.attendance_df = load_attendance_data()
        self.students_df = load_student_data()
        self.subjects_df = load_subject_data()
        self.stats_df = calculate_attendance_stats(self.attendance_df)
        
    def get_real_time_answer(self, question):
        """Generate answers for real-time queries that need fresh data."""
        question_lower = question.lower()
        
        # 1. Student ID/Enrollment Queries
        enrollment_pattern = r'(\d{14})'  # Pattern for 14-digit enrollment numbers
        enrollment_matches = re.findall(enrollment_pattern, question)
        
        if enrollment_matches:
            student_id = enrollment_matches[0]
            student = self.students_df[self.students_df['student_id'] == student_id]
            if not student.empty:
                name = student.iloc[0]['name']
                return f"The student with enrollment {student_id} is {name}"
        
        # Handle variations of enrollment/ID questions
        if ("enrollment" in question_lower or "id" in question_lower) and any(char.isdigit() for char in question):
            # Extract any number sequence that could be an ID
            numbers = re.findall(r'\d+', question)
            for number in numbers:
                student = self.students_df[self.students_df['student_id'].astype(str).str.contains(number)]
                if not student.empty:
                    name = student.iloc[0]['name']
                    student_id = student.iloc[0]['student_id']
                    return f"The student with enrollment {student_id} is {name}"
        
        # 2. Subject Code Queries
        if "subject code" in question_lower or "code" in question_lower:
            for _, subject in self.subjects_df.iterrows():
                subject_name = subject['subject_name'].lower()
                subject_abbrev = subject['subject_abbrevation'].lower()
                
                if (subject_name in question_lower or 
                    subject_abbrev in question_lower or 
                    (subject_abbrev in ['ml', 'ai', 'os'] and  # Common abbreviations
                    f" {subject_abbrev} " in f" {question_lower} ")):
                    return f"The subject code for {subject['subject_name']} ({subject['subject_abbrevation']}) is {subject['subject_code']}"
        
        # 3. Student Information Queries
        name_match = None
        for name in self.students_df['name'].unique():
            if name.lower() in question_lower:
                name_match = name
                break
        
        if name_match:
            student = self.students_df[self.students_df['name'] == name_match].iloc[0]
            
            # Phone/contact queries
            if "phone" in question_lower or "contact" in question_lower:
                return f"{name_match}'s phone number is {student['phone']}"
            
            if "email" in question_lower:
                return f"{name_match}'s email is {student['email']}"
            
            if "details" in question_lower:
                return f"Contact details for {name_match}: Email: {student['email']}, Phone: {student['phone']}"
            
            # Handle attendance queries
            student_attendance = self.attendance_df[self.attendance_df['Name'] == name_match]
            
            # Rest of existing attendance query handling code...
        
        return None
        
    def get_answer(self, question):
        """Find the most similar question and return its answer."""
        if not self.model:
            return "Chatbot model not loaded. Please run train_chatbot.py first."
        
        # First try to get a real-time answer
        real_time_answer = self.get_real_time_answer(question)
        if real_time_answer:
            return real_time_answer
        
        # Fall back to model-based answers
        question_vector = self.model['vectorizer'].transform([question])
        similarities = cosine_similarity(question_vector, self.model['question_vectors']).flatten()
        max_sim_idx = np.argmax(similarities)
        max_sim = similarities[max_sim_idx]
        
        if max_sim < 0.5:
            return self.generate_fallback_response(question)
        
        return self.model['answers'][max_sim_idx]
    
    def generate_fallback_response(self, question):
        """Generate a response for questions the model wasn't trained on."""
        question_lower = question.lower()
        
        if "present" in question_lower or "attendance" in question_lower or "attend" in question_lower:
            return "I can help you check attendance. Try asking:\n- 'Was [student name] present on [YYYY-MM-DD]?'\n- 'What is [student name]'s attendance percentage?'"
        
        if "subject" in question_lower or "course" in question_lower:
            return "I can help with subject information. Try asking:\n- 'What is the subject code for [subject]?'\n- 'How many credits is [subject] worth?'"
        
        if "contact" in question_lower or "email" in question_lower or "phone" in question_lower:
            return "I can help find student contact details. Try asking:\n- 'Give me contact details of [student name]'\n- 'What is the email of student ID [ID]?'"
        
        return "I can help with:\n1. Attendance queries\n2. Subject information\n3. Student details\n4. Analytics\nPlease rephrase your question or ask for examples."
    
    def refresh_model(self):
        """Reload both the model and data to get the latest information."""
        try:
            # Run the training script
            subprocess.run(["python", "train_chatbot.py"], check=True, capture_output=True)
            
            # Reload model and data
            self.model = self.load_model()
            self.refresh_data()
            
            return "Model and data refreshed with latest information."
        except subprocess.CalledProcessError as e:
            return f"Error refreshing model: {e.stderr.decode()}"

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance Chatbot")
        self.root.geometry("800x600")
        self.chatbot = AttendanceChatbot()
        
        # Create main frame with padding
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Attendance Management Assistant", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Create the chat display with custom styling
        self.chat_display = scrolledtext.ScrolledText(
            main_frame, 
            wrap=tk.WORD, 
            width=70, 
            height=20,
            font=("Arial", 10)
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.chat_display.tag_configure("user", foreground="#0066CC")
        self.chat_display.tag_configure("bot", foreground="#006633")
        self.chat_display.config(state=tk.DISABLED)
        
        # Create the input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create the user input field with placeholder
        self.user_input = ttk.Entry(
            input_frame, 
            width=70,
            font=("Arial", 10)
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", self.send_message)
        self.user_input.bind("<FocusIn>", lambda e: self.on_entry_click())
        self.user_input.bind("<FocusOut>", lambda e: self.on_focus_out())
        
        # Create the send button
        send_button = ttk.Button(
            input_frame, 
            text="Send",
            command=self.send_message,
            style="Accent.TButton"
        )
        send_button.pack(side=tk.RIGHT)
        
        # Create the refresh button
        refresh_button = ttk.Button(
            main_frame, 
            text="Refresh Data",
            command=self.refresh_data,
            style="Accent.TButton"
        )
        refresh_button.pack(pady=10)
        
        # Configure custom styles
        style = ttk.Style()
        style.configure("Accent.TButton", padding=5)
        
        # Set placeholder text
        self.placeholder = "Type your question here..."
        self.user_input.insert(0, self.placeholder)
        self.user_input.config(foreground='grey')
        
        # Welcome message
        self.display_bot_message(
            "Welcome! I can help you with:\n\n"
            "📅 Attendance Queries\n"
            "- Check student attendance on specific dates\n"
            "- Get attendance percentages\n"
            "- List absent students\n\n"
            "👤 Student Information\n"
            "- Get contact details\n"
            "- Check course enrollment\n\n"
            "📘 Subject Information\n"
            "- Subject codes and credits\n"
            "- Course details\n\n"
            "📊 Analytics\n"
            "- Attendance statistics\n"
            "- Students at risk\n\n"
            "How can I help you today?"
        )
    
    def on_entry_click(self):
        """Handle entry field click - clear placeholder."""
        if self.user_input.get() == self.placeholder:
            self.user_input.delete(0, tk.END)
            self.user_input.config(foreground='black')
    
    def on_focus_out(self):
        """Handle focus out - restore placeholder if empty."""
        if not self.user_input.get():
            self.user_input.insert(0, self.placeholder)
            self.user_input.config(foreground='grey')
    
    def display_bot_message(self, message):
        """Display a message from the chatbot."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "🤖 Assistant: ", "bot")
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def display_user_message(self, message):
        """Display a message from the user."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "👤 You: ", "user")
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def send_message(self, event=None):
        """Process the user's message and get a response."""
        message = self.user_input.get()
        if message == self.placeholder or message.strip() == "":
            return
        
        self.display_user_message(message)
        
        # Get response from chatbot
        response = self.chatbot.get_answer(message)
        self.display_bot_message(response)
        
        # Clear the input field
        self.user_input.delete(0, tk.END)
        # Restore placeholder
        self.on_focus_out()
    
    def refresh_data(self):
        """Refresh the chatbot model and data."""
        self.display_bot_message("Refreshing data and model... Please wait.")
        message = self.chatbot.refresh_model()
        self.display_bot_message(message)

def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
