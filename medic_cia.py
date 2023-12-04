import streamlit as st
from streamlit_chat import message as st_message
from audiorecorder import audiorecorder
import time
import json
import requests
from google.cloud import speech_v1p1beta1 as speech

# Set your Google Cloud API key or service account credentials path
google_api_key = "YOUR_GOOGLE_API_KEY_OR_SERVICE_ACCOUNT_JSON_PATH"

token_hugging_face = "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"

headers = {"Authorization": f"Bearer {token_hugging_face}"}
API_URL_DIAGNOSTIC = "https://api-inference.huggingface.co/models/abhirajeshbhai/symptom-2-disease-net"

def recognize_speech(audio_content):
    client = speech.SpeechClient.from_service_account_info(google_api_key)
    
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        return result.alternatives[0].transcript

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

def generate_answer(audio):
    st.spinner("Consultation in progress...")

    # Voice recognition model using Google Speech-to-Text API
    text = recognize_speech(audio.raw_data)

    # Disease Prediction Model
    diagnostic = diagnostic_medic(text)

    # Save conversation
    st.session_state.history.append({"message": text, "is_user": True})
    st.session_state.history.append({"message": f" Your disease would be {diagnostic}", "is_user": False})

    st.success("Medical consultation done")

if __name__ == "__main__":
    # ... (rest of your code)
    # remove the hamburger in the upper right hand corner and the Made with Streamlit footer
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
