import streamlit as st
import pickle
import re
import nltk
nltk.download("punkt")
nltk.download("stopwords")
#loading models
clf=pickle.load(open("clf.pkl", "rb"))
tfidf=pickle.load(open("tfidf.pkl", "rb"))
#web app
def main():
    st.title("Resume Screening Application")
    st.file_uploader("Upload Your Resume",type=["pdf","txt"])
    
if __name__=="__main__":
    main()