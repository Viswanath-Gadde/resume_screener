# import streamlit as st
# import pickle
# import re
# import nltk
# from PyPDF2 import PdfReader
# import re
# import string

# nltk.download("punkt")
# nltk.download("stopwords")

# # Load models
# clf = pickle.load(open("clf.pkl", "rb"))
# tfidf = pickle.load(open("tfidf.pkl", "rb"))

# # Helper function to clean resume text
# def clean(text):
#     text=re.sub("https\S+\s","",text)
#     text=re.sub("@\S","",text)
#     text=re.sub("#\S+\s","",text)
#     text=re.sub("RT|cc","",text)
#     text=re.sub('[%s]'%re.escape(string.punctuation),"",text)
#     text=re.sub('[^\x100-\x7f]'," ",text)
#     text=re.sub('\s+'," ",text)

#     return text
    

# def main():
#     st.title("Resume Screening Application")

#     uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf", "txt"])

#     if uploaded_file is not None:
#         # Read content based on file type
#         if uploaded_file.type == "application/pdf":
#             pdf_reader = PdfReader(uploaded_file)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#         else:
#             text = str(uploaded_file.read(), "utf-8")
        
#         cleaned_text = clean(text)
#         input_vector = tfidf.transform([cleaned_text])
#         pred_id = clf.predict(input_vector)[0]

#         st.write("Predicted Job ID:", pred_id)
#         category_mapping={6: 'Data Science',
#                             12: 'HR',
#                             0: 'Advocate',
#                             1: 'Arts',
#                             24: 'Web Designing',
#                             16: 'Mechanical Engineer',
#                             22: 'Sales',
#                             14: 'Health and fitness',
#                             5: 'Civil Engineer',
#                             15: 'Java Developer',
#                             4: 'Business Analyst',
#                             21: 'SAP Developer',
#                             2: 'Automation Testing',
#                             11: 'Electrical Engineering',
#                             18: 'Operations Manager',
#                             20: 'Python Developer',
#                             8: 'DevOps Engineer',
#                             17: 'Network Security Engineer',
#                             19: 'PMO',
#                             7: 'Database',
#                             13: 'Hadoop',
#                             10: 'ETL Developer',
#                             9: 'DotNet Developer',
#                             3: 'Blockchain',
#                             23: 'Testing'}
#         category_name=category_mapping.get(pred_id,"Unknown Category")
#         st.write("Predicted Job Category:", category_name)

# if __name__ == "__main__":
#     main()
import streamlit as st
import pickle
import re
import nltk
import os
from PyPDF2 import PdfReader
import string
from google import genai   # ✅ Gemini AI
from dotenv import load_dotenv  # ✅ To load API key from .env

# ====== INITIAL SETUP ======
nltk.download("punkt")
nltk.download("stopwords")

# Load environment variables safely
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except:
    client = None

# Load ML Model and TF-IDF
clf = pickle.load(open("clf.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Job Category Mapping
category_mapping = {
    6: 'Data Science', 12: 'HR', 0: 'Advocate', 1: 'Arts', 24: 'Web Designing',
    16: 'Mechanical Engineer', 22: 'Sales', 14: 'Health and fitness', 5: 'Civil Engineer',
    15: 'Java Developer', 4: 'Business Analyst', 21: 'SAP Developer',
    2: 'Automation Testing', 11: 'Electrical Engineering', 18: 'Operations Manager',
    20: 'Python Developer', 8: 'DevOps Engineer', 17: 'Network Security Engineer',
    19: 'PMO', 7: 'Database', 13: 'Hadoop', 10: 'ETL Developer',
    9: 'DotNet Developer', 3: 'Blockchain', 23: 'Testing'
}

# ====== CLEANING FUNCTION ======
def clean(text):
    text = re.sub("https\S+\s", "", text)
    text = re.sub("@\S+", "", text)
    text = re.sub("#\S+\s", "", text)
    text = re.sub("RT|cc", "", text)
    text = re.sub('[%s]' % re.escape(string.punctuation), "", text)
    text = re.sub('[^\x00-\x7f]', " ", text)
    text = re.sub('\s+', " ", text)
    return text

# ====== AI AGENT SUGGESTION FUNCTION ======
def get_resume_feedback(text, job_role):
    if not client:
        return "⚠ Gemini AI not connected. Please set your GEMINI_API_KEY."

    prompt = f"""
    You are an expert HR Resume Analyst. 
    The resume is predicted for the role: {job_role}.
    Resume Text: {text[:2000]}

    Based on this, answer:
    1. What important skills or sections are missing in this resume?
    2. What should be improved to match an ideal {job_role} resume?
    3. Give bullet-point actionable suggestions (clear and short).
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Error from AI: {e}"

# ====== MAIN STREAMLIT APP ======
def main():
    st.title("📄 AI-Powered Resume Screening & Suggestions")

    uploaded_file = st.file_uploader("Upload your Resume (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file:
        # Extract text
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() for page in pdf_reader.pages])
        else:
            text = str(uploaded_file.read(), "utf-8")

        cleaned_text = clean(text)
        input_vector = tfidf.transform([cleaned_text])
        pred_id = clf.predict(input_vector)[0]
        job_role = category_mapping.get(pred_id, "Unknown Category")

        st.success(f"✅ Predicted Job Role: **{job_role}**")

        # ===== AI Analysis Section =====
        with st.spinner("🤖 Analyzing your resume with AI..."):
            feedback = get_resume_feedback(cleaned_text, job_role)

        st.subheader("📌 AI Suggestions for Improvement")
        st.write(feedback)

if __name__ == "__main__":
    main()
