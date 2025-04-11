document.addEventListener('DOMContentLoaded', function() {
    // Get current year and set it as default
    const currentYear = new Date().getFullYear();
    document.getElementById('year').value = currentYear;
    
    // Set today's date as default
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('date').value = today;
    
    // Add event listeners
    document.getElementById('branch').addEventListener('change', handleBranchChange);
    document.getElementById('sem').addEventListener('change', handleSemesterChange);
    document.getElementById('fetchBtn').addEventListener('click', fetchAttendance);
    document.getElementById('submitBtn').addEventListener('click', submitAttendance);
    document.getElementById('selectAllBtn').addEventListener('click', selectAll);
    document.getElementById('unselectAllBtn').addEventListener('click', unselectAll);
});

function handleBranchChange() {
    const branch = document.getElementById('branch').value;
    // Clear subject dropdown
    const subjectSelect = document.getElementById('subject');
    subjectSelect.innerHTML = '<option value="">Select Subject</option>';
    
    // Load students for selected branch
    if (branch) {
        loadStudents(branch);
    } else {
        document.getElementById('studentsList').innerHTML = '';
    }
}

function handleSemesterChange() {
    const year = document.getElementById('year').value;
    const branch = document.getElementById('branch').value;
    const sem = document.getElementById('sem').value;
    
    if (year && branch && sem) {
        loadSubjects(year, branch, sem);
    }
}

function loadSubjects(year, branch, sem) {
    const subjectSelect = document.getElementById('subject');
    subjectSelect.innerHTML = '<option value="">Loading subjects...</option>';
    
    fetch(`/api/subjects?year=${year}&branch=${branch}&sem=${sem}`)
        .then(response => response.json())
        .then(subjects => {
            subjectSelect.innerHTML = '<option value="">Select Subject</option>';
            subjects.forEach(subject => {
                const option = document.createElement('option');
                option.value = subject;
                option.textContent = subject;
                subjectSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error loading subjects:', error);
            subjectSelect.innerHTML = '<option value="">Error loading subjects</option>';
        });
}

function loadStudents(branch) {
    const studentsList = document.getElementById('studentsList');
    studentsList.innerHTML = '<p>Loading students...</p>';
    
    fetch(`/api/students?branch=${branch}`)
        .then(response => response.json())
        .then(students => {
            if (students.length > 0) {
                generateStudentCheckboxes(students);
            } else {
                studentsList.innerHTML = '<p>No students found for selected branch</p>';
            }
        })
        .catch(error => {
            console.error('Error loading students:', error);
            studentsList.innerHTML = '<p>Error loading students</p>';
        });
}

function generateStudentCheckboxes(students) {
    const studentsList = document.getElementById('studentsList');
    studentsList.innerHTML = '';
    
    students.forEach(student => {
        const studentItem = document.createElement('div');
        studentItem.className = 'student-item';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `student-${student.student_id}`;
        checkbox.value = student.student_id;
        
        const label = document.createElement('label');
        label.htmlFor = `student-${student.student_id}`;
        label.textContent = `${student.student_id} - ${student.name}`;
        
        studentItem.appendChild(checkbox);
        studentItem.appendChild(label);
        studentsList.appendChild(studentItem);
    });
}

// Select all students
function selectAll() {
    const checkboxes = document.querySelectorAll('.student-item input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

// Unselect all students
function unselectAll() {
    const checkboxes = document.querySelectorAll('.student-item input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}

// Fetch attendance data
function fetchAttendance() {
    const year = document.getElementById('year').value;
    const date = document.getElementById('date').value;
    const branch = document.getElementById('branch').value;
    const sem = document.getElementById('sem').value;
    const subject = document.getElementById('subject').value;
    
    // Validate inputs
    if (!date || !branch || !sem || !subject) {
        showMessage('Please fill all required fields', 'error');
        return;
    }
    
    // Clear previous attendance data
    unselectAll();
    
    // Show loading message
    showMessage('Fetching attendance data...', '');
    
    // Fetch data from server
    fetch(`/api/attendance?year=${year}&date=${date}&branch=${branch}&sem=${sem}&subject=${subject}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.present_students && Array.isArray(data.present_students)) {
                // Mark present students
                data.present_students.forEach(studentId => {
                    // Try to find checkbox with exact student ID
                    let checkbox = document.getElementById(`student-${studentId}`);
                    if (checkbox) {
                        checkbox.checked = true;
                        console.log(`Marking student ${studentId} as present`); // Debug log
                    } else {
                        console.log(`Could not find checkbox for student ${studentId}`); // Debug log
                    }
                });
                showMessage(`Attendance data loaded successfully. Found ${data.present_students.length} present students.`, 'success');
            } else {
                showMessage('No attendance data found', '');
            }
        })
        .catch(error => {
            console.error('Error fetching attendance data:', error);
            showMessage('Error fetching attendance data', 'error');
        });
}

// Submit attendance data
function submitAttendance() {
    const year = document.getElementById('year').value;
    const date = document.getElementById('date').value;
    const branch = document.getElementById('branch').value;
    const sem = document.getElementById('sem').value;
    const subject = document.getElementById('subject').value;
    
    // Validate inputs
    if (!date || !branch || !sem || !subject) {
        showMessage('Please fill all required fields', 'error');
        return;
    }
    
    // Get selected students
    const checkboxes = document.querySelectorAll('.student-item input[type="checkbox"]:checked');
    const presentStudents = Array.from(checkboxes).map(cb => cb.value);
    
    // Prepare data to send
    const data = {
        year: year,
        date: date,
        branch: branch,
        sem: sem,
        subject: subject,
        present_students: presentStudents
    };
    
    // Show loading message
    showMessage('Submitting attendance data...', '');
    
    // Send data to server
    fetch('/api/attendance', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        showMessage('Attendance submitted successfully', 'success');
    })
    .catch(error => {
        console.error('Error submitting attendance:', error);
        showMessage('Error submitting attendance data', 'error');
    });
}

// Display status messages
function showMessage(message, type) {
    const messageElement = document.getElementById('message');
    messageElement.textContent = message;
    messageElement.className = 'message';
    
    if (type) {
        messageElement.classList.add(type);
    }
    
    // Auto-hide success messages after 3 seconds
    if (type === 'success') {
        setTimeout(() => {
            messageElement.textContent = '';
            messageElement.className = 'message';
        }, 3000);
    }
}