import requests
from pydub import AudioSegment
import streamlit as st
from streamlit_chat import message as st_message
from audiorecorder import audiorecorder
import json
import time

# Updated API details
API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
API_URL_DIAGNOSTIC = "https://api-inference.huggingface.co/models/DinaSalama/symptom_to_disease_distb"
API_URL_NEW_DIAGNOSTIC = "https://api-inference.huggingface.co/models/abhirajeshbhai/symptom-2-disease-net"
headers = {"Authorization": "Bearer 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"}

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

def diagnostic_medic(voice_text):
    payload = {"inputs": voice_text}
    response = query(payload)  # Using the existing diagnostic model

    try:
        # Extract top diseases or symptoms based on the model's output
        top_results = response[0][:5]  # Assuming the model returns a list of results
        final_output = format_diagnostic_results(top_results)
    except (KeyError, IndexError):
        final_output = 'Diagnostic information not available'

    return final_output

def query(payload):
    response = requests.post(API_URL_DIAGNOSTIC, headers=headers, json=payload)
    return response.json()

def format_diagnostic_results(results):
    try:
        if isinstance(results[0], list):
            # Assuming it's a list of results, so no need to change the format
            sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)
        elif isinstance(results[0], dict):
            # Assuming it's a dictionary, so wrap it in a list
            sorted_results = sorted([results[0]], key=lambda x: x['score'], reverse=True)
        else:
            raise TypeError("Invalid diagnostic result format")

        # Extract the names of the top 2 diseases or symptoms
        top_results = sorted_results[:2]
        formatted_results = [result['label'] for result in top_results]

        if not formatted_results:
            return 'No diagnostic information available'

        return f'Top Diseases or Symptoms:\n{", ".join(formatted_results)}'
    except (KeyError, TypeError, IndexError):
        return 'Invalid diagnostic result format'

def query_new_diagnostic_model(payload):
    response = requests.post(API_URL_NEW_DIAGNOSTIC, headers=headers, json=payload)
    return response.json()

def choose_highest_confidence(*results):
    try:
        # Flatten the results into a single list of dictionaries
        flattened_results = []

        for i, result in enumerate(results):
            if isinstance(result, list) and result:
                flattened_results.extend(result)
            elif isinstance(result, dict):
                flattened_results.append(result)

        if not flattened_results:
            raise ValueError("No valid diagnostic results found")

        # Compare confidence levels and choose the one with the highest confidence
        final_diagnostic = max(flattened_results, key=lambda x: x['score'])

        return final_diagnostic

    except (KeyError, TypeError, ValueError) as e:
        print(f"Error in choose_highest_confidence: {e}")
        print(f"Results: {results}")
        return {"error": "Invalid diagnostic result format"}

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

    try:
        # Existing Disease Prediction Model
        st.write("Calling diagnostic model...")
        diagnostic_result = diagnostic_medic(text)
        st.write(f"Diagnostic result:\n{diagnostic_result}")
    except Exception as e:
        st.error(f"Error calling the existing diagnostic model: {str(e)}")
        diagnostic_result = {"error": "Diagnostic model error"}

    try:
        # New Disease Prediction Model
        st.write("Calling new diagnostic model...")
        new_model_result = query_new_diagnostic_model({"inputs": text})
        st.write(f"New diagnostic result:\n{new_model_result}")
    except Exception as e:
        st.error(f"Error calling the new diagnostic model: {str(e)}")
        new_model_result = {"error": "New diagnostic model error"}

    try:
        # Additional Disease Prediction Model
        st.write("Calling additional diagnostic model...")
        additional_model_result = query({
            "inputs": "I like you. I love you",  # Adjust the input based on the actual requirements
        })
        st.write(f"Additional diagnostic result:\n{additional_model_result}")
    except Exception as e:
        st.error(f"Error calling the additional diagnostic model: {str(e)}")
        additional_model_result = {"error": "Additional diagnostic model error"}

    # Compare confidence levels and choose the one with higher confidence
    try:
        final_diagnostic = choose_highest_confidence(
            diagnostic_result, new_model_result, additional_model_result
        )

        st.write(f"Final diagnostic result:\n{format_diagnostic_results(final_diagnostic)}")

    except Exception as e:
        st.error(f"Error comparing diagnostic results: {str(e)}")
        final_diagnostic = {"error": "Diagnostic result comparison error"}

    # Save conversation
    st.session_state.history.append({"message": text, "is_user": True})
    st.session_state.history.append({"message": final_diagnostic, "is_user": False})

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
