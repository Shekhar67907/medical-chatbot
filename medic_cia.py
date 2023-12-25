import requests
from pydub import AudioSegment
import streamlit as st
from streamlit_chat import message as st_message
from audiorecorder import audiorecorder
import json
import time
import speech_recognition as sr  # Import the speech_recognition library

# Updated API details
API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"

# List of diagnostic models with their respective API URLs
DIAGNOSTIC_MODELS = [
    {"name": "Model 1", "api_url": "https://api-inference.huggingface.co/models/abhirajeshbhai/symptom-2-disease-net"},
    {"name": "Model 2", "api_url": "https://api-inference.huggingface.co/models/DinaSalama/symptom_to_disease_distb"},
]

headers = {"Authorization": "Bearer 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"}

def recognize_speech(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)

    try:
        # Use Google Web Speech API for recognition
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.warning("Speech recognition could not understand the audio. Please try again.")
        return "Speech recognition failed"
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Web Speech API; {e}")
        return "Speech recognition failed"
        
def diagnostic_medic(voice_text):
    model_results = []

    for model_info in DIAGNOSTIC_MODELS:
        payload = {"inputs": [voice_text]}
        response = requests.post(model_info["api_url"], headers=headers, json=payload)

        try:
            results = response.json()[0][:5]
            model_results.append({"name": model_info["name"], "results": results})
        except (KeyError, IndexError):
            st.warning(f'Diagnostic information not available for {model_info["name"]}')

    if not model_results:
        return 'No diagnostic information available'

    # Compare results based on confidentiality score and choose the model with the highest score
    best_model_result = max(model_results, key=lambda x: max([result['score'] for result in x['results']], default=0.0))
    
    return format_diagnostic_results(best_model_result["results"], best_model_result["name"])


def format_diagnostic_results(results, model_name):
    # Sort the results based on the score in descending order
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Extract the names of the top 2 diseases or symptoms
    top_results = sorted_results[:2]
    formatted_results = [result['label'] for result in top_results]

    if not formatted_results:
        return 'No diagnostic information available'

    return f'Top Diseases or Symptoms from {model_name}:\n{", ".join(formatted_results)}'

def generate_answer(audio_recording):
    st.spinner("Consultation in progress...")

    # To save audio to a file:
    audio_recording.export("audio.wav", format="wav")

    # Voice recognition model
    st.write("Audio file saved. Starting speech recognition...")
    text = recognize_speech("audio.wav")

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
        generate_answer(audio)

        for i, chat in enumerate(st.session_state.history):  # Show historical consultation
            st_message(**chat, key=str(i))
