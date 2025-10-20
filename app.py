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
import re
import nltk
import os
from PyPDF2 import PdfReader
import string
from google import genai
from dotenv import load_dotenv

# ===== INITIAL SETUP =====
nltk.download("punkt")
nltk.download("stopwords")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except:
    client = None

# ===== CLEANING FUNCTION =====
def clean(text):
    text = re.sub("https\S+\s", "", text)
    text = re.sub("@\S+", "", text)
    text = re.sub("#\S+\s", "", text)
    text = re.sub("RT|cc", "", text)
    text = re.sub('[%s]' % re.escape(string.punctuation), "", text)
    text = re.sub('[^\x00-\x7f]', " ", text)
    text = re.sub('\s+', " ", text)
    return text

# ===== AI RESUME ANALYSIS FUNCTION =====
def analyze_resume_with_ai(text):
    if not client:
        return "⚠ Gemini AI not connected. Add GEMINI_API_KEY in .env"

    prompt = f"""
    You are a professional Resume Reviewer and HR Expert.

    Analyze the following resume and answer:
    1. Key strengths and good points.
    2. Missing important sections (e.g., Projects, Skills, Experience, Education).
    3. Grammar or formatting issues.
    4. How to improve this resume to meet industry standards.
    5. Provide suggestions in bullet points.

    Resume Content:
    {text[:2500]}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ AI Error: {e}"

# ===== STREAMLIT APP =====
def main():
    st.title("📄 AI Resume Analyzer")

    uploaded_file = st.file_uploader("Upload Your Resume (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file:
        # Extract text
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() for page in pdf_reader.pages])
        else:
            text = str(uploaded_file.read(), "utf-8")

        cleaned_text = clean(text)

        # AI Analysis
        with st.spinner("🤖 Analyzing your resume..."):
            feedback = analyze_resume_with_ai(cleaned_text)

        st.subheader("✅ AI Resume Feedback & Suggestions")
        st.write(feedback)

if __name__ == "__main__":
    main()
