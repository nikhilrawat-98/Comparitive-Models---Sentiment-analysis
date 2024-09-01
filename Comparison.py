# Importing Libraries
import streamlit as st
import pdfplumber
from textblob import TextBlob
from transformers import BertTokenizer, BertForSequenceClassification, pipeline, set_seed
from transformers import RobertaTokenizer
import boto3
import PyPDF2
from io import BytesIO
import time 

# Set seed for reproducibility
set_seed(42)

# Initialize tokenizer and models explicitly for BERT and RoBERTa
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
bert_classifier = pipeline('sentiment-analysis', model=bert_model, tokenizer=bert_tokenizer)

roberta_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_classifier = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")

# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_file):
    start_time = time.time()
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    extraction_time = time.time() - start_time
    return text, extraction_time

# Function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    start_time = time.time()
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    analysis_time = time.time() - start_time
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    return sentiment, subjectivity, analysis_time

# Function to analyze sentiment using BERT
def analyze_sentiment_bert(text):
    start_time = time.time()
    results = bert_classifier(text[:512]) 
    analysis_time = time.time() - start_time
    sentiment = results[0]['label'].replace("LABEL_1", "Positive").replace("LABEL_2", "Negative").replace("LABEL_0", "Neutral")
    return sentiment, results[0]['score'], analysis_time

# Function to analyze sentiment using RoBERTa
def analyze_sentiment_roberta(text):
    label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Positive', 'LABEL_2': 'Neutral'}
    chunk_size = 500
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sentiment_scores = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    total_confidence = 0
    start_time = time.time()
    for chunk in text_chunks:
        if chunk.strip() == "":
            continue
        results = roberta_classifier(chunk)
        for result in results:
            sentiment = label_map.get(result['label'], 'Unknown')
            confidence = result['score']
            sentiment_scores[sentiment] += confidence
            total_confidence += confidence
    analysis_time = time.time() - start_time
    overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    return overall_sentiment, sentiment_scores, analysis_time

# Function to analyze sentiment using AWS Comprehend
def analyze_sentiment_comprehend(text):
    comprehend = boto3.client(
        service_name='comprehend',
        region_name='us-east-1',
        aws_access_key_id='AKIAYDJYBNVM5M7SWSA3',
        aws_secret_access_key='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'
    )

    def get_sentiment(chunk):
        return comprehend.detect_sentiment(Text=chunk, LanguageCode='en')

    chunks = []
    current_chunk = ''
    words = text.split()
    for word in words:
        if len((current_chunk + word).encode('utf-8')) + 1 <= 5000:
            current_chunk += word + ' '
        else:
            chunks.append(current_chunk)
            current_chunk = word + ' '
    if current_chunk:
        chunks.append(current_chunk)

    start_time = time.time()
    sentiments = [get_sentiment(chunk) for chunk in chunks]
    analysis_time = time.time() - start_time

    avg_sentiment = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Mixed': 0}
    for sentiment in sentiments:
        for key in avg_sentiment:
            avg_sentiment[key] += sentiment['SentimentScore'][key]
    for key in avg_sentiment:
        avg_sentiment[key] /= len(sentiments)

    dominant_sentiment = max(avg_sentiment, key=avg_sentiment.get)
    return dominant_sentiment, avg_sentiment[dominant_sentiment], analysis_time

# Streamlit app main function
def main():
    st.title("PDF Sentiment Analysis - Model Comparison")
    
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
    
    if uploaded_file is not None:
        text_content, extraction_time = extract_text_from_pdf(uploaded_file)
        st.write(f"Text extraction took {extraction_time:.2f} seconds.")
        
        if st.button('Analyze and Compare Sentiment'):
            st.write("Starting sentiment analysis...")

            # TextBlob Analysis
            tb_sentiment, tb_subjectivity, tb_time = analyze_sentiment_textblob(text_content)
            st.write(f"**TextBlob Analysis:**\nSentiment: {tb_sentiment}, Subjectivity: {tb_subjectivity:.2f}\nTotal Time: {tb_time + extraction_time + 2:.2f} seconds")

            # BERT Analysis
            bert_sentiment, bert_score, bert_time = analyze_sentiment_bert(text_content)
            st.write(f"**BERT Analysis:**\nSentiment: {bert_sentiment}, Score: {bert_score:.4f}\nTotal Time: {bert_time + extraction_time + 2:.2f} seconds")

            # RoBERTa Analysis
            roberta_sentiment, roberta_scores, roberta_time = analyze_sentiment_roberta(text_content)
            st.write(f"**RoBERTa Analysis:**\nSentiment: {roberta_sentiment}, Scores: {roberta_scores}\nTotal Time: {roberta_time + extraction_time + 2:.2f} seconds")

            # AWS Comprehend Analysis
            comprehend_sentiment, comprehend_score, comprehend_time = analyze_sentiment_comprehend(text_content)
            st.write(f"**AWS Comprehend Analysis:**\nSentiment: {comprehend_sentiment}, Score: {comprehend_score:.2f}\nTotal Time: {comprehend_time + extraction_time + 2:.2f} seconds")

if __name__ == "__main__":
    main()
