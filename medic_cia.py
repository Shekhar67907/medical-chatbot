import streamlit as st
from audiorecorder import audiorecorder  # Assuming you have a library for audio recording

import time
import json
import requests


token_hugging_face = "Your token access"

headers = {"Authorization": f"Bearer {token_hugging_face}"} #TOKEN HUGGING FACE
API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/openai/whisper-tiny.en"
API_URL_DIAGNOSTIC = "https://api-inference.huggingface.co/models/abhirajeshbhai/symptom-2-disease-net"



def recognize_speech(audio_file):
    with open(audio_file, "rb") as f:
        data = f.read()

    retries = 3
    for _ in range(retries):
        try:
            response = requests.request("POST", API_URL_RECOGNITION, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            output = json.loads(response.content.decode("utf-8"))
            final_output = output['text']
            return final_output
        except (KeyError, requests.RequestException) as e:
            print(f"Error in recognize_speech: {e}")
            time.sleep(1)

    return "Failed to recognize speech"

# Disease prediction model
def diagnostic_medic(voice_text):
    synthomps = {"inputs": voice_text}
    data = json.dumps(synthomps)

    retries = 3
    for _ in range(retries):
        try:
            response = requests.request("POST", API_URL_DIAGNOSTIC, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            output = json.loads(response.content.decode("utf-8"))
            final_output = output[0][0]['label']
            return final_output
        except (KeyError, requests.RequestException) as e:
            print(f"Error in diagnostic_medic: {e}")
            time.sleep(1)

    return "Failed to diagnose"

# Paste the recognize_speech and diagnostic_medic functions here

# Streamlit app
def main():
    st.title("Medical Diagnostic Assistant")

    # User interface for recording audio
    audio = audiorecorder("Start recording", "Recording in progress...")

    if len(audio) > 0:
        st.success("Recording complete!")

        # Voice recognition and disease prediction
        st.subheader("Analyzing... Please wait.")
        with st.spinner("Analyzing audio..."):
            text_result = recognize_speech(audio)
            disease_result = diagnostic_medic(text_result)

        # Display results
        st.subheader("Results:")
        st.write(f"Recognized Symptomps: {text_result}")
        st.write(f"Predicted Disease: {disease_result}")

if __name__ == "__main__":
    main()
