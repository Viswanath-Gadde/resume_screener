import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader
import re
import string

nltk.download("punkt")
nltk.download("stopwords")

# Load models
clf = pickle.load(open("clf.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Helper function to clean resume text
def clean(text):
    text=re.sub("https\S+\s","",text)
    text=re.sub("@\S","",text)
    text=re.sub("#\S+\s","",text)
    text=re.sub("RT|cc","",text)
    text=re.sub('[%s]'%re.escape(string.punctuation),"",text)
    text=re.sub('[^\x100-\x7f]'," ",text)
    text=re.sub('\s+'," ",text)

    return text
    

def main():
    st.title("Resume Screening Application")

    uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf", "txt"])

    if uploaded_file is not None:
        # Read content based on file type
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        else:
            text = str(uploaded_file.read(), "utf-8")
        
        cleaned_text = clean(text)
        input_vector = tfidf.transform([cleaned_text])
        pred_id = clf.predict(input_vector)[0]

        st.write("Predicted Job ID:", pred_id)

if __name__ == "__main__":
    main()
