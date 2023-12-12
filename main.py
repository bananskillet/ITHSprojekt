from main_process import main_process_start
import streamlit as st
import pandas as pd
import os
import settings

st.set_page_config(layout='wide')

def load_data(file_name):
    # Load csv-file
    folder_path = settings.DATA_RAW 
    full_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(full_path, sep=';')
    return df

def main():

    st.title('Youtube Video Summarizer')
    st.subheader('Youtube audio -> Whisper -> Language Detection -> Expand Contractions -> Sentiment Analysis -> Phraze Extraction -> Lemmatizer -> Keyword Extraction -> Stopword Removal -> OpenAI API Prompt')
    
    folder_path = settings.DATA_RAW  # Folder where csv-files are present

    # List of csv-files
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Select csv-file
    csv_file = st.selectbox('Select csv-file', csv_files)

    # Load data  
    df = load_data(csv_file)

    # Processing state
    processing = False

    # Check if url exsits
    if 'URL' in df.columns and 'Author' in df.columns and 'Title' in df.columns:
        # Create mapping
        author_title_to_url = {f"{row['Author']} - {row['Title']}": row['URL'] for index, row in df.iterrows()}
        # Select video
        selected_option = st.selectbox('Select youtube video to process', list(author_title_to_url.keys()))

        # Show processing when button is pressed 
        if not processing and st.button('Process'):
            processing = True  

            with st.spinner('Processing...'):  
                selected_csv_filepath = os.path.join(folder_path, csv_file)

                url_to_process = author_title_to_url[selected_option]
                transcribed_text, nlp_text, nlp_extented_text = main_process_start(selected_csv_filepath, url_to_process)
                
                # Display results
                st.title('Result')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text_area('Text original: Language Detection', transcribed_text, height=1000)
                with col2:
                    st.text_area('Text semi-processed: Language Detection - Expand Contractions - Sentiment Analysis - Praze Extraction - OpenAI Prompting', nlp_text, height=1000)
                with col3:
                    st.text_area('Text fully-processed: Language Detection - Expand Contractions - Sentiment Analysis - Praze Extraction - Lemmatization - Keyword Extraction - Stopword Removal - OpenAI Prompting', nlp_extented_text, height=1000)

            processing = False  

    else:
        st.error('Required columns not found in csv-file')

    st.markdown('---')
    st.markdown('Made by Daniel Andersson')

if __name__ == '__main__':
    main()





