import streamlit as st
# ----------------- Set Page Configuration -----------------
st.set_page_config(page_title="Emotion Detection AI", page_icon="🎭", layout="wide")


from model import predict_emotion, plot_emotion_probabilities
from utils import process_uploaded_file


# # Sidebar
# st.sidebar.title("📌 Model Info")
# st.sidebar.markdown("**Model Name:** `Pere-ere/fine_tuned_bert_film_sentiment_modell`")
# st.sidebar.markdown("**Trained for:** Emotion Detection in Film Scripts 🎬")
# st.sidebar.markdown("**Supports:** TXT, PDF, DOCX, CSV, XLSX")
# st.sidebar.write("---")

# Header
st.markdown(
    """
    <h1 style="text-align: center; color: #0077b6;">🎭 Emotion Detection AI</h1>
    <p style="text-align: center; font-size: 18px; color: #555;">
        Detect emotions in text using a fine-tuned BERT model. <br>
        Enter text or upload a file to analyze emotions.
    </p>
    """,
    unsafe_allow_html=True
)

# Input selection: Text input or File Upload
option = st.radio("Choose input method:", ("📄 Type Text", "📂 Upload File"), horizontal=True)

# ----------------- Text Input Section -----------------
if option == "📄 Type Text":
    user_input = st.text_area("Enter your text here:", height=150, placeholder="Type something...")
    
    if st.button("🔍 Analyze Emotion"):
        if user_input:
            (predicted_emotion, emoji), probs = predict_emotion(user_input)
            st.success(f"**Predicted Emotion:** {predicted_emotion} {emoji}")
            plot_emotion_probabilities(probs)
        else:
            st.warning("⚠️ Please enter text before analyzing.")

# ----------------- File Upload Section -----------------
elif option == "📂 Upload File":
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx", "csv", "xlsx"])
    
    if uploaded_file:
        st.write(f"✅ File uploaded: {uploaded_file.name}")  # Debugging output
        
        file_text = process_uploaded_file(uploaded_file)
        
        if file_text and "❌" not in file_text:  # Check for error message
            st.text_area("📜 Extracted Text:", file_text, height=200)
            
            if st.button("🔍 Analyze Emotion"):
                (predicted_emotion, emoji), probs = predict_emotion(file_text)
                st.success(f"**Predicted Emotion:** {predicted_emotion} {emoji}")
                plot_emotion_probabilities(probs)
        else:
            st.error(file_text)  # Display error message

# ----------------- Footer -----------------
st.markdown("<p style='text-align: center; font-size: 14px;'>Made with ❤️ using Hugging Face & Streamlit</p>", unsafe_allow_html=True)
