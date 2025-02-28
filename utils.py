import pandas as pd
import PyPDF2
import docx

def process_uploaded_file(uploaded_file):
    """Extracts text from uploaded files (TXT, PDF, DOCX, CSV, XLSX)."""
    file_type = uploaded_file.name.split(".")[-1].lower()
    
    try:
        if file_type == "txt":
            return uploaded_file.read().decode("utf-8")

        elif file_type == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            return text if text else "⚠️ No readable text found in this PDF."

        elif file_type == "docx":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text if text else "⚠️ No readable text found in this DOCX."

        elif file_type in ["csv", "xlsx"]:
            df = pd.read_csv(uploaded_file) if file_type == "csv" else pd.read_excel(uploaded_file)
            return df.to_string() if not df.empty else "⚠️ No data found in the file."

        else:
            return "❌ Unsupported file type."
    
    except Exception as e:
        return f"❌ Error processing file: {str(e)}"
