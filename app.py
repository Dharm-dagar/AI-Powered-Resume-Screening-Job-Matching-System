from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle
import os
import csv
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load models===========================================================================================================
rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))

# Hardcoded Job Description for Matching Score Calculation
JOB_DESCRIPTION = """
Role: MERN Stack Developer
- Expertise: MongoDB, Express, React, and Node.js (MERN stack).
- Front-End Proficiency: Strong skills in JavaScript, HTML, and CSS.
- RESTful APIs: Experience in designing and consuming APIs.
- Agile Development: Familiarity with agile methodologies.
- State Management: Knowledge of libraries like Redux or Context API (preferred).
- Responsive Design: Ability to build mobile-friendly, adaptive UIs.
- Performance Optimization: Experience in improving web application speed and efficiency.
- Collaboration: Comfortable working in cross-functional teams.
- Scalability: Proven ability to deliver high-performance applications.
"""

# Clean resume==========================================================================================================
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7F]+', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()
    return cleanText

# Prediction and Category Name==========================================================================================
# def predict_category(resume_text):
#     resume_text = cleanResume(resume_text)
#     resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
#     predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
#     return predicted_category

# def job_recommendation(resume_text):
#     resume_text = cleanResume(resume_text)
#     resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
#     recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
#     return recommended_job

# PDF to Text==============================================================================================================
def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Resume Parsing===========================================================================================================
def extract_contact_number_from_resume(text):
    """
    Extracts a contact number from the resume text.
    - Supports country codes (e.g., +91, +1, +44)
    - Handles various formats (e.g., (123) 456-7890, 98765 43210)
    - Detects numbers after a call icon (📞, ☎, etc.)
    - Returns the number without the country code
    """

    # Predefined list of common country codes
    country_codes = ["+1", "+44", "+91", "+61", "+49", "+33", "+81", "+86", "+7", "+55", "+39", "+34"]

    # Regex pattern to match phone numbers (including those after call icons)
    pattern = r"(?:📞|☎)?\s*(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{3,5}[-.\s]?\d{3,5}\b"

    # Search for a match
    match = re.search(pattern, text)
    
    if match:
        phone_number = match.group().strip()

        # Remove any call icon from the extracted number
        phone_number = re.sub(r"[📞☎]", "", phone_number).strip()

        # Remove any predefined country code from the extracted number
        for code in country_codes:
            if phone_number.startswith(code):
                phone_number = phone_number[len(code):].strip("- .")

        return phone_number

    return None


def extract_email_from_resume(text):
    """
    Extracts an email address from the resume text.
    - Detects emails after email icons (📧, ✉) or after "Email:"
    """
    # Regex pattern to match emails after icons or "Email:", or directly in text
    pattern = r"(?:📧|✉|Email:)\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})|([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"

    # Search for a match
    match = re.search(pattern, text)

    if match:
        email = match.group(1) or match.group(2)  # Extract the first valid email found
        return email.strip()

    return None


def extract_skills_from_resume(text):
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau', 'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib', 'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
        'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition', 'Recommendation Systems',
        'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks', 'Convolutional Neural Networks',
        'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees', 'Support Vector Machines',
        'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN', 'Association Rule Learning',
        'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL', 'Big Data Analytics',
        'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker', 'Kubernetes', 'Linux',
        'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption', 'Malware Analysis',
        'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration', 'Continuous Deployment',
        'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development', 'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite', 'Photoshop', 'Illustrator',
        'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research', 'Customer Development', 'Lean Startup',
        'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing', 'SEO', 'SEM', 'PPC',
        'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)', 'Salesforce',
        'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting', 'Ticketing Systems', 'ServiceNow'
    ]
    skills = []
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        if re.search(pattern, text, re.IGNORECASE):
            skills.append(skill)
    return skills

def extract_education_from_resume(text):
    education = []
    education_keywords = [
        'Computer Science', 'Information Technology', 'Software Engineering', 'Electrical Engineering', 'Mechanical Engineering', 'Civil Engineering',
        'Chemical Engineering', 'Biomedical Engineering', 'Aerospace Engineering', 'Nuclear Engineering', 'Industrial Engineering', 'Systems Engineering',
        'Environmental Engineering', 'Petroleum Engineering', 'Geological Engineering', 'Marine Engineering', 'Robotics Engineering', 'Biotechnology',
        'Biochemistry', 'Microbiology', 'Genetics', 'Molecular Biology', 'Bioinformatics', 'Neuroscience', 'Biophysics', 'Biostatistics', 'Pharmacology',
        'Physiology', 'Anatomy', 'Pathology', 'Immunology', 'Epidemiology', 'Public Health', 'Health Administration', 'Nursing', 'Medicine', 'Dentistry',
        'Pharmacy', 'Veterinary Medicine', 'Medical Technology', 'Radiography', 'Physical Therapy', 'Occupational Therapy', 'Speech Therapy', 'Nutrition',
        'Sports Science', 'Kinesiology', 'Exercise Physiology', 'Sports Medicine', 'Rehabilitation Science', 'Psychology', 'Counseling', 'Social Work',
        'Sociology', 'Anthropology', 'Criminal Justice', 'Political Science', 'International Relations', 'Economics', 'Finance', 'Accounting', 'Business Administration',
        'Management', 'Marketing', 'Entrepreneurship', 'Hospitality Management', 'Tourism Management', 'Supply Chain Management', 'Logistics Management',
        'Operations Management', 'Human Resource Management', 'Organizational Behavior', 'Project Management', 'Quality Management', 'Risk Management',
        'Strategic Management', 'Public Administration', 'Urban Planning', 'Architecture', 'Interior Design', 'Landscape Architecture', 'Fine Arts',
        'Visual Arts', 'Graphic Design', 'Fashion Design', 'Industrial Design', 'Product Design', 'Animation', 'Film Studies', 'Media Studies',
        'Communication Studies', 'Journalism', 'Broadcasting', 'Creative Writing', 'English Literature', 'Linguistics', 'Translation Studies',
        'Foreign Languages', 'Modern Languages', 'Classical Studies', 'History', 'Archaeology', 'Philosophy', 'Theology', 'Religious Studies',
        'Ethics', 'Education', 'Early Childhood Education', 'Elementary Education', 'Secondary Education', 'Special Education', 'Higher Education',
        'Adult Education', 'Distance Education', 'Online Education', 'Instructional Design', 'Curriculum Development'
    ]
    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        if re.search(pattern, text):
            education.append(keyword)
    return education


def extract_name_from_resume(text):
    """
    Extracts the candidate's name from a resume.
    - Checks if the name is at the top (header section).
    - Handles cases where 'Name:' is explicitly mentioned.
    - Avoids extracting job titles and other text.
    """

    # Split the text into lines and check the first few lines for a name
    lines = text.strip().split("\n")

    # Try extracting from the top lines of the document (usually the name is here)
    for line in lines[:5]:  # Checking the first few lines
        words = line.strip().split()
        if len(words) == 2 and words[0][0].isupper() and words[1][0].isupper():
            return line.strip()

    # General pattern-based extraction (Fallback)
    name_pattern = r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b"
    matches = re.findall(name_pattern, text)

    for name in matches:
        # Avoid extracting words that are job titles, skills, or generic words
        if name.lower() not in ["email", "phone", "engineer", "student", "developer", "certificate"]:
            return name.strip()

    return None

# Matching Score Calculation==============================================================================================
def calculate_matching_score(job_desc, resume_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_desc, resume_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return score

# CSV Saving Function===================================================================================================
def save_resume_data(data, filename="resume_results.csv"):
    header = ['Timestamp', 'Name', 'Phone', 'Email', 'Skills', 'Education', 'Matching Score']
    
    # Check if CSV already exists
    file_exists = os.path.isfile(filename)
    rows = []
    
    # If the file exists, read its current rows
    if file_exists:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(row)
    
    # Append the new data
    rows.append(data)
    
    # Sort rows in descending order of matching score (converted to float)
    rows.sort(key=lambda r: float(r['Matching Score']), reverse=True)
    
    # Write the sorted data back to the CSV file
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

# routes===============================================
@app.route('/')
def resume():
    return render_template("resume.html")

@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename
        if filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file.")
        
        # Process resume: predictions, extraction and matching score
        # predicted_category = predict_category(text)
        # recommended_job = job_recommendation(text)
        phone = extract_contact_number_from_resume(text) or "Not Found"
        email = extract_email_from_resume(text) or "Not Found"
        extracted_skills = extract_skills_from_resume(text)
        extracted_education = extract_education_from_resume(text)
        name = extract_name_from_resume(text) or "Not Found"
        
        matching_score = calculate_matching_score(JOB_DESCRIPTION, text)
        
        # Prepare data for CSV
        resume_data = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Name': name,
            'Phone': phone,
            'Email': email,
            'Skills': ", ".join(extracted_skills) if extracted_skills else "Not Found",
            'Education': ", ".join(extracted_education) if extracted_education else "Not Found",
            # 'Predicted Category': predicted_category,
            # 'Recommended Job': recommended_job,
            'Matching Score': matching_score
        }
        
        save_resume_data(resume_data)
        
        # Do not render extracted info on UI; return a simple confirmation message.
        return render_template('resume.html', message="Resume processed and data saved successfully.")
    else:
        return render_template("resume.html", message="No resume file uploaded.")

if __name__ == '__main__':
    app.run(debug=True)
