# frontend.py

import os
import json
import pandas as pd
import google.generativeai as genai
from datetime import date
import time
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv
import io

# --- INITIALIZATION AND CONFIGURATION ---

# Load environment variables from a .env file
load_dotenv()

# Securely get API keys and Supabase URL from environment variables
# These are configured in Streamlit's secrets manager for deployed apps
SUPABASE_URL = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
API_KEY = os.getenv('GOOGLE_API_KEY') or st.secrets.get("GOOGLE_API_KEY")

# --- Function to check for required credentials ---
def check_credentials():
    """Checks if all necessary credentials are provided."""
    if not all([SUPABASE_URL, SUPABASE_KEY, API_KEY]):
        st.error("🛑 Missing Credentials!")
        st.info("Please ensure you have set SUPABASE_URL, SUPABASE_KEY, and GOOGLE_API_KEY in your environment or Streamlit secrets.")
        st.stop()

# Call the check at the start of the script
check_credentials()

# Initialize Supabase client and configure Gemini
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=API_KEY)

# --- BACKEND LOGIC (Adapted from version1.py) ---

def get_prompt_for_extraction():
    """
    This detailed prompt tells Gemini exactly what to do: transcribe the audio
    and extract specific information into a structured JSON format.
    """
    return """
    You are a highly accurate data entry assistant for a veterinary clinic.
    Your task is to listen to the provided audio file, which is a doctor's dictation,
    and extract the following specific pieces of information.

    Format your response ONLY as a single, clean JSON object. Do not include any
    other text, explanations, or markdown formatting like ```json.

    The JSON object must have these exact keys:
    - "patient_id"
    - "patient_name"
    - "patient_dose"
    - "notes_for_doctor"

    Instructions for extraction:
    1.  **patient_id**: Extract the value associated with "Paws ID". It should be a number.
    2.  **patient_name**: Extract the value associated with "Cat name" or "Dog name".
    3.  **patient_dose**: This is critical. Combine all medications and their dosages into a single string. Separate each medication with a comma and a space. For example: "Augmentin injection 2cc, Neural fort 1cc".
    4.  **notes_for_doctor**: Extract any text that is a reminder, instruction, or observation for other staff. This often starts with "reminder for..." or "please give...". Include the full instruction.

    If any piece of information is not mentioned in the audio, use the value "N/A".
    """

def process_audio_file(uploaded_file, model):
    """
    Uploads a single audio file to the Gemini API and asks for data extraction.
    Includes retry logic for API stability.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Save the uploaded file temporarily to disk to get a path
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Upload the file to Gemini
            audio_file = genai.upload_file(path=uploaded_file.name)
            
            # Wait for processing
            while audio_file.state.name == "PROCESSING":
                time.sleep(2)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "FAILED":
                 raise Exception("File processing failed.")

            # Send to Gemini for analysis
            response = model.generate_content([get_prompt_for_extraction(), audio_file])
            
            # Clean up the response to ensure it's valid JSON
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            
            # Clean up the temporary file
            os.remove(uploaded_file.name)
            
            return json.loads(cleaned_response)

        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed for {uploaded_file.name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                st.error(f"All retries failed for {uploaded_file.name}.")
                # Clean up the temporary file on failure
                if os.path.exists(uploaded_file.name):
                    os.remove(uploaded_file.name)
                return None

# --- STREAMLIT FRONTEND ---

st.set_page_config(page_title="Veterinary Audio Processor", layout="wide")

st.title("🐾 Veterinary Audio Record Processor")
st.markdown("Upload audio dictations from doctors to automatically extract patient information, dosages, and notes.")

# File Uploader
st.header("1. Upload Audio Files")
st.info("You can upload multiple audio files at once (e.g., MP3, WAV, M4A).")
uploaded_files = st.file_uploader(
    "Drag and drop or click to upload audio files",
    type=['mp3', 'wav', 'm4a', 'ogg', 'flac','opus'],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"Successfully uploaded {len(uploaded_files)} file(s).")
    for file in uploaded_files:
        st.audio(file, format=file.type)

# Processing Button and Logic
st.header("2. Process and View Results")
if st.button("✨ Process Audio Files", type="primary", disabled=not uploaded_files):
    
    # Use Gemini 1.5 Pro model
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    all_records = []
    
    progress_bar = st.progress(0, text="Starting processing...")

    for i, file in enumerate(uploaded_files):
        progress_text = f"Processing file: {file.name}..."
        progress_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
        
        with st.spinner(progress_text):
            extracted_data = process_audio_file(file, model)
        
        if extracted_data:
            record = {
                "patient_id": int(extracted_data.get("patient_id", 0)),
                "patient_name": extracted_data.get("patient_name", "N/A"),
                "patient_dose": extracted_data.get("patient_dose", "N/A"),
                "notes_for_doctor": extracted_data.get("notes_for_doctor", "N/A"),
                "record_date": date.today().strftime('%Y-%m-%d'),
            }
            try:
                sb.table("veterinary_records").insert(record).execute()
                st.write(f"✅ Success: Data inserted into Supabase for {file.name}.")
            except Exception as e:
                st.error(f"❌ Error inserting into Supabase for {file.name}: {e}")
            all_records.append(record)
        else:
            st.error(f"❌ Failed: Could not process {file.name}.")

    progress_bar.empty()

    if all_records:
        st.header("3. Extracted Records")
        
        df = pd.DataFrame(all_records)
        
        # Rename and reorder columns for display
        final_df = df.rename(columns={
            'patient_id': 'Patient ID',
            'patient_name': 'Patient Name',
            'patient_dose': 'Patient Dose',
            'notes_for_doctor': 'Notes for Doctor',
            'record_date': 'Date'
        })
        final_df = final_df[['Patient ID', 'Patient Name', 'Patient Dose', 'Notes for Doctor', 'Date']]
        
        st.dataframe(final_df, use_container_width=True)
        
        # Provide a download button for the results as an Excel file
        @st.cache_data
        def convert_df_to_excel(df):
            output = io.BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            return output.getvalue()

        excel_data = convert_df_to_excel(final_df)
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_data,
            file_name="veterinary_records.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.warning("Always double-check the extracted dosages and patient information against the source audio.", icon="⚠️")
    else:
        st.info("No data was successfully extracted from the uploaded files.")