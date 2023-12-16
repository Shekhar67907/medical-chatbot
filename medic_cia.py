import requests
from pydub import AudioSegment
import streamlit as st
from streamlit_chat import message as st_message
from audiorecorder import audiorecorder
import json
import time

token_hugging_face = "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"

headers = {"Authorization": f"Bearer {token_hugging_face}"}
API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
API_URL_DIAGNOSTIC = "https://api-inference.huggingface.co/models/runaksh/Symptom-2-disease_distilBERT"

# ... (Previous code remains unchanged)

def diagnostic_medic(voice_text):
    payload = {"inputs": voice_text}
    
    response = requests.post(API_URL_DIAGNOSTIC, headers=headers, json=payload)

    if response.status_code == 503:  # HTTP 503 Service Unavailable
        estimated_time = response.json().get('estimated_time', 50.0)
        st.warning(
            f"Model is currently loading. Please wait for approximately {estimated_time:.2f} seconds and try again.")
        time.sleep(estimated_time)
        return diagnostic_medic(voice_text)  # Retry after waiting

    if response.status_code != 200:
        st.error(f"Diagnostic API error: {response.content}")
        return "Diagnostic prediction failed"

    output = response.json()
    
    try:
        # Extract top diseases or symptoms based on the model's output
        top_results = output[0][:5]  # Assuming the model returns a list of results
        final_output = format_diagnostic_results(top_results)
    except (KeyError, IndexError):
        final_output = 'Diagnostic information not available'

    return final_output

# ... (Rest of the code remains unchanged)

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
