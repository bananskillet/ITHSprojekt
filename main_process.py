import os
import whisper
from pydub import AudioSegment
from pytube import YouTube
import settings
import pandas as pd
import datetime
from langdetect import detect
import contractions
import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from keybert import KeyBERT
#nltk.download('stopwords')
from nltk.corpus import stopwords
from openai_call import generate_summarization

# Define folder names
data_temp_audio = settings.DATA_TEMP_AUDIO
data_transcribed_text = settings.DATA_TRANSCRIBED_TEXT
whisper_load_model = settings.WHISPER_LOAD_MODEL
lengt_of_summarization = settings.LENGTH_OF_OUTPUT_SUMMARIZATION

# 1 
def download_youtube_audio(url, filename):
        print('Downloading audio....')
        # Initialize YouTube object
        yt = YouTube(url)

        # Select the audio stream with the highest bitrate
        audio_stream = yt.streams.filter(only_audio=True).order_by('bitrate').desc().first()
        
        # Download the audio stream and save it with the random name
        audio_stream.download(output_path=data_temp_audio, filename=filename)
    
        return 

# 2
def transcribe_audio(filename, filename_ogg, segment_length=30000):
    print('Transcribing audio...')

    audiofile = os.path.join(settings.DATA_TEMP_AUDIO, filename)

    # Split to segments
    audio = AudioSegment.from_file(audiofile)
    length = len(audio)
    segments = [audio[i:i+segment_length] for i in range(0, length, segment_length)]

    # Transcribe segments
    model = whisper.load_model(settings.WHISPER_LOAD_MODEL)
    combined_text = ''
    temp_ogg_file = os.path.join(settings.DATA_TEMP_AUDIO,filename_ogg)
    for i, segment in enumerate(segments):
        segment.export(temp_ogg_file, format='ogg')
        result = model.transcribe(temp_ogg_file, fp16=False)
        combined_text += result['text'] + ' '
        os.remove(temp_ogg_file)  # Clean up temporary file
  
    # Detect language
    language = detect(combined_text)
    print(f'Detected language: {language}')

    # Change dir and remove extension
    filename_rem_ext = filename[:-4]
    print('Saving transcribed textfile...')
    
    target_file_path = os.path.join(data_transcribed_text, filename_rem_ext + '.txt')
    with open(target_file_path, mode='wt', encoding='utf-8') as f:
        f.write(combined_text)
        f.close()

    return combined_text, language


# 3 
def delete_audio_file(filename):
    print('Deleting audio file...')
    audio_file_path = os.path.join(data_temp_audio, filename)
    
    if os.path.exists(audio_file_path):
        
        os.remove(audio_file_path)
    return

# 4 
def expand_contractions(text):
     print('Expanding text...')
     expanded_text = contractions.fix(text)
     return expanded_text

# 5
def sentiment_analysis(text):
    print('Sentiment analysis...')
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)

    compound_score = score['compound']

    if compound_score >= settings.SENTIMENT_POSITIVE_SCORE:
        return 'Positive'
    elif compound_score <= settings.SENTIMENT_NEGATIVE_SCORE:
        return 'Negative'
    else:
        return 'Neutral'
# 6
def phrase_extraction(text):
    print('Extracting phrases...')

    rake_nltk_var = Rake()

    rake_nltk_var.extract_keywords_from_text(text)
    phrases_extracted = rake_nltk_var.get_ranked_phrases()

    phrases = '; '.join(phrases_extracted[:settings.NUMBER_OF_PHRASES])

    return phrases

# 7 
def word_lemmatizer(text):
    print('Lemmatizing text...')
    lemmatizer = WordNetLemmatizer()

    # Using a RegexpTokenizer to handle alphanumeric words
    tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')

    words = word_tokenize(text)

    lemmatized_words = []
    for word in words:
        if word.isalpha():  # Check if the token is a word
            lemmatized_word = lemmatizer.lemmatize(word)
            lemmatized_words.append(lemmatized_word)
        else:  # If the token is punctuation, attach it to the previous word
            lemmatized_words[-1] += word

    lemmatized_text = ' '.join(lemmatized_words)

    return lemmatized_text

# 8 
def keyword_extraction(text):
    print('Keywords extracting...')
    # Init KeyBERT
    kw_model = KeyBERT(model=settings.KEYWORD_EXTRACTION_MODEL)

    # Extract keywords with their scores
    keywords_with_scores = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=settings.NUMBER_OF_KEYWORDS)

    # Sort keywords by score in descending order
    keywords_sorted = sorted(keywords_with_scores, key=lambda x: x[1], reverse=True)

    # Extract just the keywords and join them into a single string
    keywords_joined = ';'.join([keyword for keyword, score in keywords_sorted])

    return keywords_joined

# 9
def stopword_removal(text):
    print('Removing stopwords...')

    stop_words = set(stopwords.words('english'))

    tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')

    # Tokenize the text
    words = tokenizer.tokenize(text)

    # Remove stopwords and handle punctuation
    filtered_words = []
    for word in words:
        if word.isalpha():  
            if word.lower() not in stop_words:
                filtered_words.append(word)
        else:  
            if filtered_words:
                filtered_words[-1] += word

    # Joining the filtered words back into a sentence
    stopword_filtered_text = ' '.join(filtered_words)

    return stopword_filtered_text
 
# Start workflow 
def main_process_start(file_path, url):
    print('Begin')
    df = pd.read_csv(file_path, sep=';')

    filtered_df = df[df['URL'] == url]

    for index, row in filtered_df.iterrows():
        
        url = row['URL']
        title = row['Title']
        filename = row['Video_ID'] + '.mp4'
        filename_txt_ext = row['Video_ID'] + '.txt'
        filename_ogg = row['Video_ID'] + '.ogg'
    
        # 1 Download audio from youtube 
        download_youtube_audio(url, filename)

        # 2 Transcribe audio, extract language and save text to file
        transcribed_text, language = transcribe_audio(filename, filename_ogg)

        # 3 Delete audio file for lower disk space 
        delete_audio_file(filename)

        # 4 Transform contractions
        expanded_text = expand_contractions(transcribed_text)
        
        # 5 Detect sentiment
        sentiment = sentiment_analysis(expanded_text)
        
        # 6 Extract phrases from text and save text to folder data_nlp and save text in a seperate variable 
        phrases = phrase_extraction(expanded_text)
        print(f'Phrases extracted: {phrases}')

        # 7 Lemmatize text
        lemmatized_text = word_lemmatizer(expanded_text)

        # 8 Extract keywords
        keywords = keyword_extraction(lemmatized_text)
        print(f'keywords extraxted: {keywords}')

        # 9 Remove stopwords and save text to folder data_nlp_extended
        stopword_filtered_text = stopword_removal(lemmatized_text)

        # 10 (Send two different texts to openai)
        print('Generating summarizations...')

        nlp_text = generate_summarization(expanded_text, title, phrases, keywords, sentiment, lengt_of_summarization)
        nlp_extented_text = generate_summarization(stopword_filtered_text, title, phrases, keywords, sentiment, lengt_of_summarization)


        current_timestamp = datetime.datetime.now()

        # New data for csv
        df.at[index, 'Language'] = language
        df.at[index, 'Textfile_Original'] = filename_txt_ext
        df.at[index, 'Sentiment'] = sentiment
        df.at[index, 'Phrases'] = phrases
        df.at[index, 'Keywords'] = keywords
        df.at[index, 'Timestamp'] = current_timestamp

    # Save the updated df data to csv
    print('Updating csv file with new values...')
    df.to_csv(file_path, sep=';', index=False)
    print('Completed')

    return transcribed_text, nlp_text, nlp_extented_text