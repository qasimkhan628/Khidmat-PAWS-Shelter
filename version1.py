import os
import json
import pandas as pd
import google.generativeai as genai
from datetime import date
import time

# --- CONFIGURATION ---

# Securely get the API key from environment variables
# IMPORTANT: Make sure you have set the GOOGLE_API_KEY environment variable
API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError("API Key not found. Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=API_KEY)

# Folder where your audio files are stored
AUDIO_FOLDER = "audio_files"

# Name for the output Excel file
OUTPUT_EXCEL_FILE = "veterinary_records.xlsx"

# --- SCRIPT LOGIC ---

def create_audio_folder_if_not_exists():
    """Creates the audio folder if it doesn't exist."""
    if not os.path.exists(AUDIO_FOLDER):
        print(f"Creating directory '{AUDIO_FOLDER}'. Please place your audio files here.")
        os.makedirs(AUDIO_FOLDER)

def get_prompt_for_extraction():
    """
    This is the heart of the operation. This detailed prompt tells Gemini exactly
    what to do: transcribe the audio and extract specific information into a
    structured JSON format.
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

def process_audio_file(file_path, model):
    """
    Uploads a single audio file to the Gemini API and asks for data extraction.
    Includes retry logic for API stability.
    """
    print(f"  - Uploading {os.path.basename(file_path)}...")
    
    # Gemini 1.5 Pro can handle large files, but for robustness, we add retries.
    max_retries = 3
    for attempt in range(max_retries):
        try:
            audio_file = genai.upload_file(path=file_path)
            # Wait for the file to be processed
            while audio_file.state.name == "PROCESSING":
                time.sleep(2)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "FAILED":
                 raise Exception("File processing failed.")

            print(f"  - Sending to Gemini for analysis...")
            response = model.generate_content([get_prompt_for_extraction(), audio_file])
            
            # Clean up the response to ensure it's valid JSON
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            
            return json.loads(cleaned_response)

        except Exception as e:
            print(f"    - Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait before retrying
            else:
                print(f"    - All retries failed for {os.path.basename(file_path)}.")
                return None

def main():
    """Main function to orchestrate the entire process."""
    print("--- Veterinary Audio Processing Script ---")
    
    create_audio_folder_if_not_exists()

    # Use Gemini 2.5 Pro, which is the latest powerful model for this task
    model = genai.GenerativeModel('models/gemini-2.5-pro')
    
    all_records = []
    
    audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac'))]

    if not audio_files:
        print(f"\nNo audio files found in the '{AUDIO_FOLDER}' directory. Exiting.")
        return

    print(f"\nFound {len(audio_files)} audio file(s) to process.")

    for filename in audio_files:
        print(f"\nProcessing file: {filename}...")
        file_path = os.path.join(AUDIO_FOLDER, filename)
        
        extracted_data = process_audio_file(file_path, model)
        
        if extracted_data:
            # Add the current date to the record
            extracted_data['date'] = date.today().strftime('%Y-%m-%d')
            all_records.append(extracted_data)
            print(f"  - Success: Data extracted for {filename}.")
        else:
            print(f"  - Failed: Could not process {filename}.")

    if not all_records:
        print("\nNo data was successfully extracted from any audio files.")
        return

    print(f"\nProcessing complete. Extracted data from {len(all_records)} file(s).")

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(all_records)
    
    # Reorder and rename columns to the desired format
    final_df = df.rename(columns={
        'patient_id': 'Patient ID',
        'patient_name': 'Patient Name',
        'patient_dose': 'Patient Dose',
        'notes_for_doctor': 'Notes for doctor',
        'date': 'Date'
    })
    
    # Ensure the column order is exactly as requested
    final_df = final_df[['Patient ID', 'Patient Name', 'Patient Dose', 'Notes for doctor', 'Date']]

    # Save the DataFrame to an Excel file
    try:
        final_df.to_excel(OUTPUT_EXCEL_FILE, index=False)
        print(f"\nSuccessfully saved all records to '{OUTPUT_EXCEL_FILE}'.")
    except Exception as e:
        print(f"\nError saving to Excel file: {e}")

    print("\n--- IMPORTANT: PLEASE VERIFY ALL DATA ---")
    print("AI is a tool to assist, not replace, professional diligence.")
    print("Always double-check the extracted dosages and patient information against the source audio.")


if __name__ == "__main__":
    main()
