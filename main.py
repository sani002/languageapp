import os
import streamlit as st
from banglaspeech2text import Speech2Text
from dotenv import load_dotenv
import google.generativeai as genai
import time

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the speech-to-text model
available_models = Speech2Text.list_models()
stt = Speech2Text("base", cache_path="cache", use_gpu=True)

# Initialize Google Generative AI model (Gemini Pro)
model = genai.GenerativeModel('gemini-pro')

st.write('# Upload Audio for Transcription and AI Processing')

# Audio input for transcription
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
if uploaded_file is not None:
    # Temporarily save the uploaded file to process
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Perform speech-to-text on the saved audio file
    text = stt.recognize(audio_path)
    str1 = "The Bengali string that is given here was transcribed from audio to text. It has errors here and there and lacks punctuation. Read it and try to understand what it is trying to say, then rewrite it properly in English (don't change valuable numbers or medical details in the process):"
    result = f"{str1} {text}"

    # Display the transcribed text
    st.write("## Transcribed Text")
    st.text_area("Transcription", value=text, height=150, help="This is the raw transcription from the audio file.")

    # Send message to AI and display response
    response = model.generate_content(result)
    st.write("## mBot Response")
    message_placeholder = st.empty()
    full_response = ''
    for chunk in response.text.split(' '):  # Assuming response.text returns a string
        full_response += chunk + ' '
        time.sleep(0.05)  # Adjust time as needed for desired effect
        message_placeholder.text(full_response + '▌')
    message_placeholder.text(full_response)
