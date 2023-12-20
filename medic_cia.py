import requests
from pydub import AudioSegment
import streamlit as st
from audiorecorder import audiorecorder
import time

class APIConfig:
    # API details for Speech Recognition
    RECOGNITION_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
    RECOGNITION_HEADERS = {"Authorization": "Bearer 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"}

    # API details for Disease Prediction Models
    DIAGNOSTIC_URL_1 = "https://api-inference.huggingface.co/models/abhirajeshbhai/symptom-2-disease-net"
    DIAGNOSTIC_URL_2 = "https://api-inference.huggingface.co/models/DinaSalama/symptom_to_disease_distb"
    DIAGNOSTIC_HEADERS = {"Authorization": "Bearer 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"}

def handle_api_response(response, retry_function, *args):
    if response.status_code == 503:  # HTTP 503 Service Unavailable
        estimated_time = response.json().get('estimated_time', 20.0)
        st.warning(
            f"Model is currently loading. Please wait for approximately {estimated_time:.2f} seconds and try again.")
        time.sleep(estimated_time)
        return retry_function(*args)  # Retry after waiting

    return response.json()

def recognize_speech(audio_file):
    with open(audio_file, "rb") as f:
        data = f.read()

    response = requests.post(APIConfig.RECOGNITION_URL, headers=APIConfig.RECOGNITION_HEADERS, data=data)

    if response.status_code != 200:
        st.error(f"Speech recognition API error: {response.content}")
        return {"text": "Speech recognition failed"}

    return handle_api_response(response, recognize_speech, audio_file)

def query_diagnostic_with_retry(payload, api_url):
    response = requests.post(api_url, headers=APIConfig.DIAGNOSTIC_HEADERS, json=payload)
    return handle_api_response(response, query_diagnostic_with_retry, payload, api_url)

def format_diagnostic_results(results):
    if results and isinstance(results, list):
        sorted_results = sorted(results, key=lambda x: x.get('score', 0) if isinstance(x, dict) else 0, reverse=True)
        formatted_results = [result.get('label', 'Unknown Disease') if isinstance(result, dict) and 'label' in result else 'Unknown Disease' for result in sorted_results[:2]]
        return f'Top Diseases:\n{", ".join(formatted_results)}'

    return 'No diagnostic information available'

def diagnostic_medic(voice_text):
    payload = {"inputs": voice_text}

    # Query the first diagnostic model
    response_1 = query_diagnostic_with_retry(payload, APIConfig.DIAGNOSTIC_URL_1)
    confidence_1 = response_1[0].get('score', 0.0) if response_1 else 0.0

    # Query the second diagnostic model
    response_2 = query_diagnostic_with_retry(payload, APIConfig.DIAGNOSTIC_URL_2)
    confidence_2 = response_2[0].get('score', 0.0) if response_2 else 0.0

    # Compare confidence scores and determine the final diagnostic result
    if confidence_1 > confidence_2:
        final_results = response_1
    else:
        final_results = response_2

    # Extract the symptoms with their confidence levels
    symptoms_with_confidence = [(result.get('label', 'Unknown Symptom'), result.get('score', 0.0)) for result in final_results] if final_results else []

    # Sort symptoms by confidence level in descending order
    sorted_symptoms = sorted(symptoms_with_confidence, key=lambda x: x[1], reverse=True)

    # Extract the symptom with the highest confidence
    top_symptom, top_confidence = sorted_symptoms[0] if sorted_symptoms else ('Unknown Symptom', 0.0)

    return f'Highest Confidence Symptom: {top_symptom} (Confidence: {top_confidence:.2f})'

def generate_answer(audio_recording):
    st.spinner("Consultation in progress...")

    audio_recording.export("audio.flac", format="flac")

    st.write("Audio file saved. Starting speech recognition...")
    recognition_result = recognize_speech("audio.flac")

    # Check if speech recognition was successful
    if "recognition failed" in recognition_result.get('text', '').lower():
        st.error("Voice recognition failed. Please try again.")
        return

    # Extract the transcribed text from the recognition result
    text = recognition_result.get('text', 'Unknown Text')

    st.write(f"Speech recognition result: {text}")

    st.write("Calling diagnostic models...")
    diagnostic = diagnostic_medic(text)
    st.write(f"Diagnostic result:\n{diagnostic}")

    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.extend([
        {"message": text, "is_user": True},
        {"message": diagnostic, "is_user": False}
    ])

    st.success("Medical consultation done")

if __name__ == "__main__":
    st.title("Medical Diagnostic Assistant")
    audio = audiorecorder("Start recording", "Recording in progress...")
    generate_answer(audio)
