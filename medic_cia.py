from pydub import AudioSegment
import streamlit as st
from streamlit_chat import message as st_message
from audiorecorder import audiorecorder
import json
import requests
import time  # Added for retry delay

token_hugging_face = "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"

headers = {"Authorization": f"Bearer {token_hugging_face}"}
API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
API_URL_DIAGNOSTIC = "https://api-inference.huggingface.co/models/BillyTK616/distillbert-finetuned-medical-symptoms"

def recognize_speech(audio_file):
    with open(audio_file, "rb") as f:
        data = f.read()

    response = requests.post(API_URL_RECOGNITION, headers=headers, data=data)

    if response.status_code == 503:  # HTTP 503 Service Unavailable
        estimated_time = response.json().get('estimated_time', 50.0)
        st.warning(f"Model is currently loading. Please wait for approximately {estimated_time:.2f} seconds and try again.")
        time.sleep(estimated_time)
        return recognize_speech(audio_file)  # Retry after waiting

    if response.status_code != 200:
        st.error(f"Speech recognition API error: {response.content}")
        return "Speech recognition failed"

    output = response.json()
    final_output = output.get('text', 'Speech recognition failed')
    return final_output

def diagnostic_medic(voice_text):
    synthomps = {"inputs": voice_text}
    data = json.dumps(synthomps)

    response = requests.post(API_URL_DIAGNOSTIC, headers=headers, data=data)
    output = response.json()

    try:
        final_output = output[0][0]['label']
    except (KeyError, IndexError):
        final_output = 'Diagnostic information not available'

    return final_output

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
    st.write("Calling diagnostic model...")
    diagnostic = diagnostic_medic(text)
    st.write(f"Diagnostic result: {diagnostic}")

    # Save conversation
    st.session_state.history.append({"message": text, "is_user": True})
    st.session_state.history.append({"message": f" Your disease would be {diagnostic}", "is_user": False})

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
