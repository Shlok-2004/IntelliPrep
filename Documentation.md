# 9.1 Installation and Uninstallation

The following section outlines the complete, step-by-step procedure for setting up the IntelliPrep application on a local development environment, as well as instructions for safely removing the application from your system.

## 9.1.1 System Prerequisites

Before beginning the installation process, ensure that your system meets the following minimum requirements:
- **Operating System:** Windows 10/11, macOS (Catalina or newer), or a modern Linux distribution (e.g., Ubuntu 20.04+).
- **Python:** Python 3.10 is highly recommended (Python 3.8+ is the minimum requirement).
- **Hardware Requirements:** At least 4GB RAM is required (8GB recommended) due to the use of Machine Learning models (BERT, MediaPipe, OpenCV). A webcam is required for the HR video analysis round.
- **Git:** Version control system to clone the repository.
- **Database:** Access to a PostgreSQL instance (either installed locally or hosted remotely, such as on NeonDB).

For Ubuntu/Debian Linux users, specific system libraries are required for OpenCV. Run the following command in your terminal before starting the Python setup:
```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

## 9.1.2 Step-by-Step Installation Guide

### Step 1: Clone the Source Code Repository
Begin by downloading the project source code to your local machine using Git. Open your terminal or command prompt and execute:
```bash
git clone <repository_url>
cd intelliprep
```
This will create a new directory named `intelliprep` containing all project files.

### Step 2: Set Up a Virtual Environment (Highly Recommended)
To prevent dependency conflicts with other Python projects on your system, it is strongly advised to install IntelliPrep within a dedicated virtual environment.

**For Windows Users (Command Prompt / PowerShell):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**For macOS and Linux Users:**
```bash
python3 -m venv venv
source venv/bin/activate
```
Once activated, you should see `(venv)` prefixed to your terminal prompt.

### Step 3: Install Core Python Dependencies
With the virtual environment activated, install the required packages using the provided `requirement.txt` file. This file contains specific versions of Flask, Scikit-learn, Sentence-Transformers, MediaPipe, OpenCV, and other essential libraries.
```bash
pip install -r requirement.txt
```
*Note: Depending on your internet connection, this step may take several minutes as it downloads large machine learning frameworks.*

### Step 4: Install the NLP Model (Spacy)
The resume parsing and question evaluation engines rely on Spacy's small English core web model. Install it explicitly using the following command:
```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

### Step 5: Database Configuration
IntelliPrep utilizes PostgreSQL for user management, session tracking, and storing evaluation results. 

1. Create a `.env` file in the root directory of the project (the same folder as `app.py`).
2. Add your database connection string to this file. If you are using NeonDB or another hosted service, it will look like this:
```env
DATABASE_URL=postgresql://<username>:<password>@<host_url>/<database_name>?sslmode=require
```
3. Ensure that your database schema is initialized. You can do this by executing the SQL scripts provided in the `database/interview_prep_db.sql` file against your PostgreSQL instance.

### Step 6: Verify Dataset Availability
The machine learning question selection algorithms require local datasets. Ensure that the following CSV files exist in the root directory:
- `questions.csv`
- `Technical.csv`
- `hr.csv`
- `apptitude.csv`

### Step 7: Launch the Application
You are now ready to run the application. Start the Flask development server by executing:
```bash
python app.py
```
Alternatively, you can run it using Gunicorn (recommended for production environments):
```bash
gunicorn --bind 0.0.0.0:8000 --timeout 120 app:app
```
Open your web browser and navigate to `http://127.0.0.1:5000` (or `http://localhost:8000` if using Gunicorn) to access the IntelliPrep platform.

---

## 9.1.3 Alternative Installation via Docker

If you prefer containerization to avoid configuring Python and system dependencies manually, you can use Docker.

1. Ensure **Docker Desktop** (or Docker Engine) is installed and running on your system.
2. Build the Docker image from the root directory:
```bash
docker build -t intelliprep-app .
```
3. Run the container, passing the database environment variable:
```bash
docker run -p 8000:8000 --env-file .env intelliprep-app
```
Access the application at `http://localhost:8000`.

---

## 9.1.4 Uninstallation Guide

To completely remove the IntelliPrep application and its associated components from your system, follow these steps safely:

### 1. Stop the Running Application Local Server
- If you are running the application via `python app.py` or `flask run`, go to the terminal window running the server and stop it by pressing `Ctrl + C`.
- If you are running the application via Docker, list your active containers to find the Container ID:
  ```bash
  docker ps
  ```
- Stop and remove the container:
  ```bash
  docker stop <container_id>
  docker rm <container_id>
  ```
- Optionally, remove the Docker image to free up disk space:
  ```bash
  docker rmi intelliprep-app
  ```

### 2. Deactivate the Virtual Environment
If you were using a Python virtual environment, deactivate it to return to your global system environment:
```bash
deactivate
```

### 3. Delete the Project Files directory
Navigate to the parent directory of the `intelliprep` project folder and delete the entire folder.

**On Windows (Command Prompt):**
```cmd
rmdir /S /Q intelliprep
```
**On macOS/Linux (Terminal):**
```bash
rm -rf intelliprep
```
This will remove the source code, the virtual environment, the locally downloaded machine learning models, and the datasets.

### 4. Remove the Database (Optional)
If you created a local PostgreSQL database specifically for IntelliPrep and no longer need it, you should drop it to free up space. Open your PostgreSQL CLI (`psql`) or a graphical tool like pgAdmin, and execute:
```sql
DROP DATABASE <database_name>;
```

---
<br><br>

# 9.2 User Manual

## 9.2.1 Introduction to IntelliPrep
Welcome to IntelliPrep, an AI-powered mock interview and resume evaluation platform. IntelliPrep is designed to bridge the gap between candidates and corporate expectations by providing realistic interview simulations. Using cutting-edge Machine Learning (ML), Natural Language Processing (NLP), and Computer Vision techniques, the platform assesses your technical knowledge, aptitude skills, and behavioral traits, offering actionable feedback to help you secure your dream job.

This user manual provides a comprehensive, step-by-step guide to navigating the platform, utilizing its core features, and interpreting the analytical feedback provided.

---

## 9.2.2 Account Setup and Authentication

To access the personalized features of IntelliPrep, you must create an account. The platform uses a secure session-based authentication system to track your individual progress over time.

### Registering a New Account
1. Open your web browser and navigate to the IntelliPrep URL.
2. Click on the **Sign Up** button located in the top navigation bar or the main landing page.
3. You will be redirected to the registration form. Enter your **Full Name**, a valid **Email Address**, and a secure **Password**.
4. Click the **Register** button. If the email is not already in use, your account will be successfully created, and you will be redirected to the Login page.

### Logging In
1. Click on the **Login** link in the navigation menu.
2. Enter the registered **Email** and **Password**.
3. Upon successful authentication, you will be redirected to your personalized **Dashboard**.

### Password Recovery
If you forget your password:
1. Click the **Forgot Password?** link on the Login page.
2. Enter your registered email address and your new desired password.
3. The system will securely update your credentials, allowing you to log in immediately with the new password.

---

## 9.2.3 The User Dashboard

The Dashboard acts as your central command center. It leverages data visualizations to provide a high-level summary of your readiness for real-world interviews.

- **Welcome Banner:** Greets you with your registered name.
- **KPI Cards:** Three primary metrics are displayed at the top of the dashboard:
  - **Total Interviews:** The total number of mock interview sessions you have completed.
  - **Average Score:** Your overall aggregate score across all attempted interview questions (Technical, HR, and Aptitude), presented as a percentage.
  - **Latest Resume Score:** The Applicant Tracking System (ATS) compatibility score of the most recent resume you analyzed.
- **Score Distribution Chart (Pie Chart):** A visual breakdown of your average performance segregated by question types (e.g., how you perform in HR rounds compared to Technical rounds).
- **Recent Activity Table:** A chronological log of your last four interview questions, detailing the question category, the date attempted, and the final score received.

---

## 9.2.4 Conducting a Mock Interview

The Interview Practice module is the core feature of IntelliPrep. It intelligently selects questions based on your targeted job role and dynamically evaluates your answers.

### Configuring the Session
1. Navigate to the **Interview** tab from the main menu.
2. **Select Job Role:** Choose the specific role you are practicing for from the dropdown menu (e.g., Data Scientist, Data Analyst, Full Stack Developer, Software Engineer).
3. **Select Question Type:** Choose the domain you want to test:
   - *Technical:* Domain-specific theoretical and practical questions.
   - *Aptitude:* Logical reasoning and quantitative multiple-choice questions.
   - *HR:* Behavioral and situational questions evaluating your personality and communication.
4. **Number of Questions:** Select how many questions you want in this session (e.g., 5, 10).
5. Click **Start Interview**.

### Answering Questions (Technical & Aptitude)
1. The platform will present the first question on the screen.
2. If it is a **Technical** question, read it carefully and type your comprehensive answer into the provided text box.
3. If it is an **Aptitude** question, it will be presented with Multiple Choice Options (A, B, C, D). Type the correct option or the exact matching text into the text box.
4. Click **Submit Answer**.
5. The NLP engine will instantly evaluate your response against the ideal answer using Semantic Similarity (BERT embeddings) and Keyword Matching (TF-IDF). 
6. You will immediately receive a **Score (0-100%)** and **Detailed Feedback** highlighting what you did well and what critical concepts you missed.
7. Click **Next** to proceed to the subsequent question until the session concludes.

### Answering Questions (HR Video Round)
The HR round utilizes your webcam to simulate a face-to-face behavioral interview.
1. Ensure you are in a well-lit room and your webcam is active.
2. The HR behavioral question will be displayed on the screen.
3. Click the **Start Recording** button. Speak clearly into the camera, maintaining eye contact and professional posture.
4. Once you have finished answering, click **Stop Recording**.
5. Click **Submit Video**. 
6. The system will securely process the video using MediaPipe and TensorFlow models. It evaluates:
   - **Textual Context:** What you actually said.
   - **Facial Expressions:** Detection of confidence, nervousness, or dominant emotions.
   - **Eye Contact & Posture:** Consistency in engaging with the camera.
7. You will receive a blended score and feedback emphasizing both your communication content and your non-verbal body language.

---

## 9.2.5 Resume Analyzer Engine

The Resume Analyzer helps you tailor your CV for specific job descriptions to bypass automated Applicant Tracking Systems (ATS) used by modern recruiters.

1. Navigate to the **Resume** tab.
2. **Upload CV:** Click the upload target and select your resume file. The system accepts standard document formats (`.pdf` and `.docx`).
3. **Target Job Description (JD):** Paste the exact job description of the role you are applying for into the provided text area.
4. Click the **Analyze Resume** button.
5. The system extracts the text from your document and matches it against the JD utilizing role-specific profile dictionaries.

**Interpreting the Results:**
- **Overall ATS Score:** A percentage indicating how well your resume matches the job requirements. Aim for a score above 80% before applying.
- **Matched Skills:** A list of critical keywords successfully found in your resume.
- **Missing Skills:** A list of crucial technologies or concepts mentioned in the JD that are absent from your resume.
- **Actionable Feedback:** Auto-generated advice suggesting specific sections of your resume to rewrite or skills to acquire.

---

## 9.2.6 Progress Tracking and Analytics

To monitor your improvement over consistent practice, IntelliPrep provides a dedicated Progress page.

1. Navigate to the **Progress** tab from the main navigation menu.
2. **Radar Analytics:** The Radar Chart outlines your proficiency across the three pillars: HR, Aptitude, and Technical. A skewed shape indicates a specific weakness (e.g., excellent technical skills but poor HR performance) that requires attention.
3. **Daily Trend Graph:** A Line Chart displaying your average interview scores day-by-day. Use this to verify that your scores are trending upward as you practice more.
4. **Historical Session Log:** A comprehensive, paginated table at the bottom of the screen containing your entire history on the platform. You can review the exact date, the role you applied for, the question type, the score achieved, and most importantly, the historical feedback provided by the AI, allowing you to revisit past mistakes and learn from them.

---

## Conclusion
By consistently utilizing the Interview Practice simulations, refining your CV through the Resume Analyzer, and monitoring your weaknesses via the Progress Dashboard, IntelliPrep ensures you walk into your next real-world interview with maximum confidence and data-backed preparation.
