import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
from docx import Document
from fpdf import FPDF
import re

# ----------------- Helper Functions -----------------

def extract_text(file):
    """Extract text from PDF or DOCX."""
    if file.type == "application/pdf":
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return ""

def split_sections(text):
    """Split resume text into sections."""
    sections = {}
    current_sec = "Other"
    sections[current_sec] = ""
    lines = text.split("\n")
    for line in lines:
        header = line.strip().upper()
        if header in ["SUMMARY", "EXPERIENCE", "SKILLS", "PROJECTS", "EDUCATION"]:
            current_sec = header
            sections[current_sec] = ""
        else:
            sections[current_sec] += line + "\n"
    return sections

def analyze_section(section_name, section_text, analyzer):
    """Generate AI feedback for a section."""
    if not section_text.strip():
        return f"{section_name} is missing or empty."
    
    prompt = f"""You are an expert resume reviewer. Analyze the {section_name} section.
    Provide concise, actionable feedback. Suggest improvements with measurable achievements, keywords, and impact-oriented language."""
    
    response = analyzer(section_text + "\n" + prompt, max_length=200)[0]['generated_text']
    return response

def generate_feedback(text, analyzer):
    sections = split_sections(text)
    feedback = {}
    for sec, content in sections.items():
        feedback[sec] = analyze_section(sec, content, analyzer)
    return feedback

def get_role_keywords(role, analyzer):
    """Generate role-specific ATS keywords using AI."""
    if not role:
        return []
    prompt = f"List 10–15 important resume keywords for a {role} role. Provide only keywords, separated by commas."
    response = analyzer(prompt, max_length=60)[0]['generated_text']
    keywords = [kw.strip() for kw in response.split(",") if kw.strip()]
    return keywords

def check_ats_keywords(text, keywords):
    if not keywords:
        return [], []
    present = [kw for kw in keywords if re.search(rf"\b{kw}\b", text, re.I)]
    missing = [kw for kw in keywords if kw not in present]
    return present, missing

def create_pdf(feedback, filename="Optimized_Resume.pdf"):
    """Generate a PDF with all feedback sections."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Optimized Resume Feedback", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 12)
    for sec, fb in feedback.items():
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 8, sec, ln=True)
        pdf.set_font("Arial", '', 12)
        for line in fb.split("\n"):
            if line.strip():
                pdf.multi_cell(0, 6, "- " + line.strip())
        pdf.ln(3)
    
    pdf.output(filename)
    return filename

# ----------------- Streamlit UI -----------------

st.title("AI Resume Optimizer – Section-wise Feedback + PDF Export")

uploaded_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])
role_input = st.text_input("Target Role (optional)", "")

if uploaded_file:
    st.info("Extracting text from resume...")
    resume_text = extract_text(uploaded_file)
    st.success("Text extraction completed!")
    
    st.info("Analyzing resume sections... This may take 15-20 seconds on first run.")
    analyzer = pipeline("text2text-generation", model="google/flan-t5-base")
    
    # Section-wise feedback
    feedback = generate_feedback(resume_text, analyzer)
    
    st.subheader("Section-wise Feedback")
    for sec, fb in feedback.items():
        st.markdown(f"**{sec}**")
        st.write(fb)
    
    # Role-specific ATS keywords
    role_keywords = get_role_keywords(role_input, analyzer) if role_input else []
    present, missing = check_ats_keywords(resume_text, role_keywords)
    
    st.subheader("ATS Keywords Check")
    if role_input:
        st.write(f"Target Role: **{role_input}**")
        st.write(f"✅ Present Keywords: {', '.join(present) if present else 'None'}")
        st.write(f"❌ Missing Keywords: {', '.join(missing) if missing else 'None'}")
    else:
        st.write("Enter a target role to see ATS keyword suggestions.")
    
    # Overall suggestions for missing sections
    st.subheader("Overall Suggestions")
    if "SUMMARY" not in feedback or "missing" in feedback.get("SUMMARY", "").lower():
        st.write("- Add a concise summary with measurable achievements.")
    if "EXPERIENCE" not in feedback or "missing" in feedback.get("EXPERIENCE", "").lower():
        st.write("- Include detailed experience bullets with metrics.")
    if "SKILLS" not in feedback or "missing" in feedback.get("SKILLS", "").lower():
        st.write("- Clearly list programming languages, tools, and relevant platforms.")
    if "PROJECTS" not in feedback:
        st.write("- Consider adding relevant projects to highlight practical skills.")
    if "EDUCATION" not in feedback:
        st.write("- Include your educational background for completeness.")
    
    # PDF Export
    st.subheader("Download Optimized Resume Feedback PDF")
    if st.button("Generate PDF"):
        pdf_file = create_pdf(feedback)
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="Optimized_Resume.pdf",
                mime="application/pdf"
            )
        st.success("PDF generated successfully!")
