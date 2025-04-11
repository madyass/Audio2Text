import streamlit as st
from transformers import pipeline

device = "cpu" #select device


def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    
    #set the pipeline
    pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device = device,
    ) 

    return pipe

def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    ner_pipeline = pipeline("ner" , model = "dslim/bert-base-NER"  , aggregation_strategy="simple")
    return ner_pipeline


def transcribe_audio(uploaded_file, pipe):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
    Returns:
        str: Transcribed text from the audio file.
    """
    
    uploaded_file = uploaded_file.read() #model do not accept streamlit.uploaded.file.type , therefore i convert it to text
    prediction = pipe(uploaded_file , return_timestamps=True)["text"] # return_timestamps=True argument provide process audio which is larger than 30s
    return prediction
    


def extract_entities(text, ner_pipeline):
    """
    Extract entities from transcribed text using the NER model.
    Args:
        text (str): Transcribed text.
        ner_pipeline: NER pipeline loaded from Hugging Face.
    Returns:
        dict: Grouped entities (ORGs, LOCs, PERs).
    """
    entities = ner_pipeline(text) 
    
    #create lists
    per_list = []
    org_list = []
    loc_list = []
    
    #create dictionary 
    grouped_entities = {"PERs": per_list, "ORGs": org_list, "LOCs": loc_list}

    for entity in entities:
        word = entity['word']
        entity_type = entity.get('entity_group' , entity.get('entity')) #aggreation_stragety="simple" argument cause that ner model returns 'entity_group' 
                                                                        #if there is not 'entity_group' it returns 'entity'
        
        if "PER" in entity_type and word not in grouped_entities['PERs']:  # person
            grouped_entities["PERs"].append(word)
        elif "ORG" in entity_type and word not in grouped_entities['ORGs']: # organization
            grouped_entities["ORGs"].append(word)
        elif "LOC" in entity_type and word not in grouped_entities['LOCs']:  # location
            grouped_entities["LOCs"].append(word)

    return grouped_entities


def main():
    st.title("Meeting Transcription and Entity Extraction")

    st.write("Upload a businees meeting audio file to:")
    st.write("1. Transcribe the meeting audio into text.")
    st.write("2. Extract key entities such as Persons,Organizations,Dates and Locations")

    uploaded_file = st.file_uploader("Upload a audio file(WAV format)", type=["wav"])

    if uploaded_file is not None:
        
        #load models
        transcribe_model = load_whisper_model()
        ner_pipeline = load_ner_model()

        #transcribe model
        st.markdown( #streamlit have not a command to color background , but streamlit allow html and css for editing. I used html and css to edit.
            """
                <div style="background-color: #1A4D71; padding: 5px; border-radius: 6px;">
                    <p style="font-size: 12px; margin-top : 10px;">Transcribing the audio file... This may take a minute</p>
                </div>
             """, 
            unsafe_allow_html=True)
        
        text = transcribe_audio(uploaded_file , transcribe_model) #get transcribed text

        #show text 
        st.write("### Transcription")
        st.write(text)

        st.markdown(
            """
                <div style="background-color: #1A4D71; padding: 5px; border-radius: 6px;">
                    <p style="font-size: 12px; margin-top : 10px;">Extracting entities...</p>
                </div>
             """, 
            unsafe_allow_html=True)
                   
        grouped_entities = extract_entities(text, ner_pipeline) #extract entities

        #write entities
        st.write("### Extracted Entities")
        
        st.write("#### Organizations (ORGs):")
        st.markdown("\n".join([f"- {word}" for word in grouped_entities["ORGs"]]))
        
        st.write("#### Locations (LOCs):")
        st.markdown("\n".join([f"- {word}" for word in grouped_entities["LOCs"]]))

        st.write("#### Persons (PERs):")
        st.markdown("\n".join([f"- {word}" for word in grouped_entities["PERs"]]))
        


if __name__ == "__main__":
    main()
