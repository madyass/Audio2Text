# 🎙️ Meeting Transcription and Entity Extraction App

This Streamlit web application allows users to upload an audio file (in WAV format) from a business meeting. The app automatically transcribes the audio into text using OpenAI's Whisper model and then extracts named entities (like persons, organizations, and locations) from the transcribed text using a BERT-based NER model.

---

Features

-  Upload and transcribe business meeting audio files
-  Automatically extract named entities (Persons, Organizations, Locations)
-  Built with HuggingFace Transformers and Streamlit
-  Simple and clean user interface

---

##Models Used

- **Speech Recognition**: [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny)  
  - Lightweight and fast automatic speech recognition model
- **Named Entity Recognition (NER)**: [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER)  
  - BERT-based model fine-tuned for extracting entities like `PER`, `ORG`, and `LOC`

---

##Installation

Make sure you have Python 3.8+ installed.  
Install the dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
