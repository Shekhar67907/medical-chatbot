import requests
from pydub import AudioSegment
import streamlit as st
from streamlit_chat import message as st_message
from audiorecorder import audiorecorder
import json
import time

API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
API_URL_DIAGNOSTIC = "https://api-inference.huggingface.co/models/shanover/medbot_godel_v3"

DIAGNOSTIC_MODELS = [
    {"name": "Model 1", "api_url": "https://api-inference.huggingface.co/models/abhirajeshbhai/symptom-2-disease-net"},
    {"name": "Model 2", "api_url": "https://api-inference.huggingface.co/models/DinaSalama/symptom_to_disease_distb"},
]

headers = {"Authorization": "Bearer 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"}

def query_diagnostic(payload):
    response = requests.post(API_URL_DIAGNOSTIC, headers=headers, json=payload)
    return response.json()

def recognize_speech(audio_file):
    with open(audio_file, "rb") as f:
        data = f.read()

    response = requests.post(API_URL_RECOGNITION, headers=headers, data=data)

    if response.status_code == 503:  # HTTP 503 Service Unavailable
        estimated_time = response.json().get('estimated_time', 20.0)
        st.warning(
            f"Model is currently loading. Please wait for approximately {estimated_time:.2f} seconds and try again.")
        time.sleep(20)
        return recognize_speech(audio_file)  # Retry after waiting

    if response.status_code != 200:
        st.error(f"Speech recognition API error: {response.content}")
        return "Speech recognition failed"

    output = response.json()
    final_output = output.get('text', 'Speech recognition failed')
    return final_output

def get_precautions(disease):
    # Define precautions for each disease
    precautions_dict = {
        "peptic_ulcer": ["Drink plenty of water.", "Avoid spicy and acidic foods.", "Take prescribed medications."],
        # Add more diseases and their precautions as needed
    }

    # Check if precautions are available for the predicted disease
    precautions = precautions_dict.get(disease.lower())

    if precautions:
        return precautions
    else:
        return ["Precautions not available for this disease."]

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

    # Extract the names and scores of the top results
    top_results = sorted_results[:2]
    formatted_results = [(result['label'], result['score']) for result in top_results]

    if not formatted_results:
        return 'No diagnostic information available'

    # Create a string with disease names and confidence scores
    formatted_results_str = ', '.join([f'{label} ({score:.2%})' for label, score in formatted_results])

    return f'Top Diseases or Symptoms from {model_name}:\n{formatted_results_str}'
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
    diagnostic_result = query_diagnostic({"inputs": text})

    # Check if the response is a list and get the first element
    if isinstance(diagnostic_result, list) and diagnostic_result:
        diagnostic_result = diagnostic_result[0]

    # Extract the predicted disease
    predicted_disease = diagnostic_result.get('output', 'Unknown').lower() if isinstance(diagnostic_result, dict) else 'Unknown'

    st.write(f"Predicted Disease: {predicted_disease}")

    # Get precautions for the predicted disease
    precautions = get_precautions(predicted_disease)

    # Display precautions
    st.write("Precautions:")
    for precaution in precautions:
        st.write(f"- {precaution}")

    # Add the statement for more detailed symptoms
    st.write("Please provide more detailed symptoms for precise recognition.")

    # Save conversation
    st.session_state.history.append({"message": text, "is_user": True})
    st.session_state.history.append({"message": f"Predicted Disease: {predicted_disease}", "is_user": False})
    st.session_state.history.append({"message": "Precautions:", "is_user": False})
    for precaution in precautions:
        st.session_state.history.append({"message": f"- {precaution}", "is_user": False})

    st.success("Medical consultation done")

if __name__ == "__main__":
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
