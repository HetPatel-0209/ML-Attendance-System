import os
import datetime

def create_directory_structure():
    """Create the initial directory structure for the attendance system"""
    # Get current year
    current_year = datetime.datetime.now().year
    
    # Base directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/attendanceData", exist_ok=True)
    os.makedirs("faces", exist_ok=True)
    
    # Year directory
    year_dir = f"data/attendanceData/{current_year}"
    os.makedirs(year_dir, exist_ok=True)
    
    # Programs
    programs = ["IT", "CP", "CSD"]
    for program in programs:
        program_dir = f"{year_dir}/{program}"
        os.makedirs(program_dir, exist_ok=True)
        
        # Semesters
        for sem in range(1, 9):
            sem_dir = f"{program_dir}/sem{sem}"
            os.makedirs(sem_dir, exist_ok=True)
    
    print(f"Directory structure created successfully in {year_dir}")
    
    # Create students.csv if it doesn't exist
    if not os.path.exists("data/students.csv"):
        with open("data/students.csv", "w") as f:
            f.write("student_id,name,course,email,phone\n")
        print("Created empty students.csv file")

if __name__ == "__main__":
    create_directory_structure()
    print("Setup completed successfully. You can now run main.py to start the application.") 