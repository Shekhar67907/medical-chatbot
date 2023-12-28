import requests
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
import streamlit as st
from streamlit_chat import message as st_message
from audiorecorder import audiorecorder
import json
import time
import os

# Updated API details
API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"

# List of diagnostic models with their respective API URLs
DIAGNOSTIC_MODELS = [
    {"name": "Model 1", "api_url": "https://api-inference.huggingface.co/models/abhirajeshbhai/symptom-2-disease-net"},
    {"name": "Model 2", "api_url": "https://api-inference.huggingface.co/models/DinaSalama/symptom_to_disease_distb"},
]

headers = {"Authorization": "Bearer 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"}

def preprocess_audio(audio_file):
    # Load audio file
    audio = AudioSegment.from_file(audio_file)

    # Reduce noise using low_pass_filter with a cutoff frequency of 3000 Hz (you can adjust this value)
    audio = low_pass_filter(audio, 3000)

    # Normalize amplitude
    audio = normalize(audio)

    return audio

def recognize_speech(audio):
    # Save AudioSegment to a temporary file
    temp_audio_file = "temp_audio.wav"
    audio.export(temp_audio_file, format="wav")

    # Read the temporary file as binary data
    with open(temp_audio_file, "rb") as f:
        audio_data = f.read()

    # Make a bytes-like object
    audio_bytes = bytes(audio_data)

    response = requests.post(
        API_URL_RECOGNITION,
        headers=headers,
        data=audio_bytes,
    )

    if response.status_code == 503:  # HTTP 503 Service Unavailable
        estimated_time = response.json().get('estimated_time', 50.0)
        st.warning(
            f"Model is currently loading. Please wait for approximately {estimated_time:.2f} seconds and try again.")
        time.sleep(20)
        return recognize_speech(audio)  # Retry after waiting

    if response.status_code != 200:
        st.error(f"Speech recognition API error: {response.content}")
        return "Speech recognition failed"

    output = response.json()
    final_output = output.get('text', 'Speech recognition failed')

    # Clean up temporary file
    os.remove(temp_audio_file)

    return final_output

def diagnostic_medic(voice_text):
    # ... (unchanged)

def format_diagnostic_results(results, model_name):
    # ... (unchanged)

def generate_answer(audio_recording):
    st.spinner("Consultation in progress...")

    # To save audio to a file:
    audio_recording.export("audio.wav", format="wav")

    # Preprocess audio
    preprocessed_audio = preprocess_audio("audio.wav")

    # Voice recognition model
    st.write("Audio file saved. Starting speech recognition...")
    text = recognize_speech(preprocessed_audio)

    if "recognition failed" in text.lower():
        st.error("Voice recognition failed. Please try again.")
        return

    st.write(f"Speech recognition result: {text}")

    # Disease Prediction Model
    st.write("Calling diagnostic models...")
    diagnostic = diagnostic_medic(text)
    st.write(f"Diagnostic result:\n{diagnostic}")

    # Add the statement for more detailed symptoms
    st.write("Please provide more detailed symptoms for precise recognition.")

    # Save conversation
    st.session_state.history.append({"message": text, "is_user": True})
    st.session_state.history.append({"message": diagnostic, "is_user": False})

    st.success("Medical consultation done")

# ... (rest of your code remains the same)

if __name__ == "__main__":
    # Remove the hamburger in the upper right-hand corner and the Made with Streamlit footer
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image("./logo_.png", width=200)

    with col3:
        st.write(' ')

    if "history" not in st.session_state:
        st.session_state.history = []

    st.title("Medical Diagnostic Assistant")

    # Show Input
    audio = audiorecorder("Start recording", "Recording in progress...")

    if audio:
         print("Calling generate_answer...")
         generate_answer(audio)
         print("generate_answer called")

         for i, chat in enumerate(st.session_state.history):  # Show historical consultation
             st_message(**chat, key=str(i))
