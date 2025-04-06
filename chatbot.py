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

class AttendanceChatbot:
    def __init__(self):
        self.model = self.load_model()
        
    def load_model(self):
        """Load the trained chatbot model."""
        try:
            with open('models/chatbot_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            print("Model file not found. Please run train_chatbot.py first.")
            return None
        
    def get_answer(self, question):
        """Find the most similar question and return its answer."""
        if not self.model:
            return "Chatbot model not loaded. Please train the model first."
        
        # Transform the question using the vectorizer
        question_vector = self.model['vectorizer'].transform([question])
        
        # Calculate similarity with all stored questions
        similarities = cosine_similarity(question_vector, self.model['question_vectors']).flatten()
        
        # Find the most similar question
        max_sim_idx = np.argmax(similarities)
        max_sim = similarities[max_sim_idx]
        
        # If similarity is too low, it's probably a question we can't answer
        if max_sim < 0.5:
            return self.generate_fallback_response(question)
        
        return self.model['answers'][max_sim_idx]
    
    def generate_fallback_response(self, question):
        """Generate a response for questions the model wasn't trained on."""
        # Check for common attendance-related keywords
        question_lower = question.lower()
        
        if "present" in question_lower or "attendance" in question_lower or "attend" in question_lower:
            return "I don't have specific information about that attendance query. Try asking about a specific student or date."
        
        if "course" in question_lower or "class" in question_lower:
            return "I can tell you about student courses if you specify a student name."
        
        if "who" in question_lower and "missing" in question_lower:
            return "You can ask 'Who was present on [date]?' and I can provide that information."
        
        if "excel" in question_lower or "file" in question_lower or "sheet" in question_lower:
            return "Attendance is being recorded in ML.xlsx located in data/attendanceData/2025/IT/sem6 folder."
        
        return "I'm not sure how to answer that. I can help with attendance queries like 'Was [student] present on [date]?' or 'Who was present on [date]?'"
    
    def refresh_model(self):
        """Reload the model to get the latest data."""
        # Run the training script to update the model with latest data
        try:
            subprocess.run(["python", "train_chatbot.py"], check=True, capture_output=True)
            self.model = self.load_model()
            return "Model refreshed with the latest attendance data."
        except subprocess.CalledProcessError as e:
            return f"Error refreshing model: {e.stderr.decode()}"

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance Chatbot")
        self.root.geometry("600x500")
        self.chatbot = AttendanceChatbot()
        
        # Create the main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the chat display
        self.chat_display = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=60, height=20)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.chat_display.config(state=tk.DISABLED)
        
        # Create the input frame
        input_frame = ttk.Frame(main_frame, padding="5")
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create the user input field
        self.user_input = ttk.Entry(input_frame, width=50)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", self.send_message)
        
        # Create the send button
        send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_button.pack(side=tk.RIGHT)
        
        # Create the refresh button
        refresh_button = ttk.Button(main_frame, text="Refresh Data", command=self.refresh_data)
        refresh_button.pack(pady=10)
        
        # Welcome message
        self.display_bot_message("Welcome to the Attendance Chatbot! You can ask questions like:\n" +
                               "- Was [student name] present on [date]?\n" +
                               "- Who was present on [date]?\n" +
                               "- What course is [student name] enrolled in?\n\n" +
                               "Attendance is being recorded in ML.xlsx in the data/attendanceData/2025/IT/sem6 folder.")
    
    def display_bot_message(self, message):
        """Display a message from the chatbot."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "Chatbot: " + message + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def display_user_message(self, message):
        """Display a message from the user."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "You: " + message + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def send_message(self, event=None):
        """Process the user's message and get a response."""
        message = self.user_input.get()
        if message.strip() == "":
            return
        
        self.display_user_message(message)
        
        # Get response from chatbot
        response = self.chatbot.get_answer(message)
        self.display_bot_message(response)
        
        # Clear the input field
        self.user_input.delete(0, tk.END)
    
    def refresh_data(self):
        """Refresh the chatbot model with latest data."""
        self.display_bot_message("Refreshing model with latest attendance data... This may take a moment.")
        message = self.chatbot.refresh_model()
        self.display_bot_message(message)

def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
