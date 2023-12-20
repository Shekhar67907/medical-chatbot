import requests
from pydub import AudioSegment
import streamlit as st
from streamlit_chat import message as st_message
from audiorecorder import audiorecorder
import time

# Updated API details for Speech Recognition
API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
headers = {"Authorization": "Bearer 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"}

# API details for Disease Prediction Models
API_URL_DIAGNOSTIC_1 = "https://api-inference.huggingface.co/models/abhirajeshbhai/symptom-2-disease-net"
API_URL_DIAGNOSTIC_2 = "https://api-inference.huggingface.co/models/DinaSalama/symptom_to_disease_distb"
headers_diagnostic = {"Authorization": "Bearer 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"}

def recognize_speech(audio_file):
    with open(audio_file, "rb") as f:
        data = f.read()

    response = requests.post(API_URL_RECOGNITION, headers=headers, data=data)

    if response.status_code == 503:  # HTTP 503 Service Unavailable
        estimated_time = response.json().get('estimated_time', 50.0)
        st.warning(
            f"Model is currently loading. Please wait for approximately {estimated_time:.2f} seconds and try again.")
        time.sleep(estimated_time)
        return recognize_speech(audio_file)  # Retry after waiting

    if response.status_code != 200:
        st.error(f"Speech recognition API error: {response.content}")
        return "Speech recognition failed"

    output = response.json()
    final_output = output.get('text', 'Speech recognition failed')
    return final_output

def query_diagnostic(payload, api_url):
    response = requests.post(api_url, headers=headers_diagnostic, json=payload)
    return response.json()

def format_diagnostic_results(results):
    # Check if results is not empty
    if results:
        # Assuming results is a list of dictionaries
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)

        # Extract the names of the top 2 diseases
        top_results = sorted_results[:2]
        formatted_results = [result.get('label', 'Unknown Disease') for result in top_results]

        return f'Top Diseases:\n{", ".join(formatted_results)}'

    return 'No diagnostic information available'

def diagnostic_medic(voice_text):
    payload = {"inputs": voice_text}

    # Query the first diagnostic model
    response_1 = query_diagnostic(payload, API_URL_DIAGNOSTIC_1)
    try:
        if isinstance(response_1, list):
            top_results_1 = response_1
        else:
            top_results_1 = response_1.get('predictions', [])

        confidence_1 = top_results_1[0]['score']
    except (KeyError, IndexError):
        confidence_1 = 0.0

    # Query the second diagnostic model
    response_2 = query_diagnostic(payload, API_URL_DIAGNOSTIC_2)
    try:
        if isinstance(response_2, list):
            top_results_2 = response_2
        else:
            top_results_2 = response_2.get('predictions', [])

        confidence_2 = top_results_2[0]['score']
    except (KeyError, IndexError):
        confidence_2 = 0.0

    # Compare confidence scores and determine the final diagnostic result
    if confidence_1 > confidence_2:
        final_results = top_results_1
    else:
        final_results = top_results_2

    return format_diagnostic_results(final_results)

def generate_answer(audio_recording):
    st.spinner("Consultation in progress...")

    # To save audio to a file:
    audio_recording.export("audio.flac", format="flac")  # Save as FLAC for Whisper ASR

    # Voice recognition model
    st.write("Audio file saved. Starting speech recognition...")
    text = recognize_speech("audio.flac")  # Use Whisper ASR for speech recognition

    if "recognition failed" in text.lower():
        st.error("Voice recognition failed. Please try again.")
        return

    st.write(f"Speech recognition result: {text}")

    # Disease Prediction Model
    st.write("Calling diagnostic models...")
    diagnostic = diagnostic_medic(text)
    st.write(f"Diagnostic result:\n{diagnostic}")

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
