import streamlit as st
from transformers import pipeline
import PyPDF2
import docx
from fpdf import FPDF
import io

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="AI Resume Optimizer", layout="wide")
st.title("AI Resume Optimizer (Free)")
st.write("Upload your resume to get instant ATS-friendly feedback and role-specific optimization.")

# -----------------------------
# Resume Upload Section
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

resume_text = ""
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = read_pdf(uploaded_file)
    else:
        resume_text = read_docx(uploaded_file)
    st.text_area("Extracted Resume Text", resume_text, height=300)

# -----------------------------
# Target Job Role Input
# -----------------------------
target_role = st.text_input("Enter Target Job Title (e.g., Data Engineer)")

# -----------------------------
# Hugging Face Model Pipeline
# -----------------------------
if resume_text:
    reviewer = pipeline("text2text-generation", model="google/flan-t5-base")

# -----------------------------
# AI Feedback on Resume
# -----------------------------
if resume_text:
    st.subheader("AI Feedback on Resume")
    feedback = reviewer(
        "You are an expert resume reviewer. Analyze this resume and suggest improvements in bullet points:\n" 
        + resume_text,
        max_length=256,
        do_sample=False
    )
    st.write(feedback[0]['generated_text'])

# -----------------------------
# Role-Specific Resume Optimization
# -----------------------------
tailored = None
if target_role and resume_text:
    st.subheader(f"Optimized Resume for {target_role}")
    tailored = reviewer(
        f"Rewrite this resume for the role of {target_role}. "
        "Provide structured output with sections: Summary, Experience, Skills. "
        "Include measurable achievements and relevant keywords:\n" + resume_text,
        max_length=500,
        do_sample=False
    )
    st.text_area("Tailored Resume (Structured)", tailored[0]['generated_text'], height=400)

# -----------------------------
# Download as TXT
# -----------------------------
if tailored:
    if st.button("Download as TXT"):
        buffer = io.BytesIO()
        buffer.write(tailored[0]['generated_text'].encode())
        st.download_button(
            label="Download TXT",
            data=buffer,
            file_name="optimized_resume.txt",
            mime="text/plain"
        )

# -----------------------------
# Download as PDF
# -----------------------------
if tailored:
    if st.button("Download as PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in tailored[0]['generated_text'].split("\n"):
            pdf.multi_cell(0, 8, txt=line)
        pdf_output = "optimized_resume.pdf"
        pdf.output(pdf_output)
        with open(pdf_output, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name=pdf_output,
                mime="application/pdf"
            )
