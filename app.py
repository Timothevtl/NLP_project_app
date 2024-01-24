import streamlit as st
from gensim.models import Word2Vec
from transformers import pipeline
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import requests
import joblib
import os
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO
import zipfile
import random

def get_english_stopwords():
    nltk.download('stopwords')
    english_stopwords = set(stopwords.words('english'))
    return english_stopwords

def clean_text(text):
    english_stopwords = get_english_stopwords()
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    # Remove stopwords
    filtered_words = [word for word in words if word not in english_stopwords]
    return " ".join(filtered_words)

def transform_query(query, tfidf_vectorizer):
    query_tfidf = tfidf_vectorizer.transform([query])
    return query_tfidf

def find_similar_books(query,tfidf_vectorizer, tfidf_matrix, book_names, book_summaries,book_rating, top_n):
    query_tfidf = transform_query(query, tfidf_vectorizer)
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    return [(book_names[i], cosine_similarities[i], book_summaries[i], book_rating[i]) for i in related_docs_indices]

def download_and_unzip(url, zip_path, extract_to='.'):
    # Check if zip file already exists
    if not os.path.exists(zip_path):
        # Download the zip file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                f.write(response.content)
        else:
            response.raise_for_status()

    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Function to predict sentiment
def predict_sentiment(review, model, vectorizer, label_encoder):
    review = clean_text(review)
    review_vectorized = vectorizer.transform([review])
    prediction_label = model.predict(review_vectorized)[0]
    prediction_proba = model.predict_proba(review_vectorized)[0]
    prediction_index = label_encoder.transform([prediction_label])[0]
    score = prediction_proba[prediction_index]
    return prediction_label, score

def download_file_from_github(file_url, file_name):
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Failed to download file from {file_url}")

def load_model_from_github(file_name, github_url):
    if not os.path.exists(file_name):
        download_file_from_github(github_url + file_name, file_name)
    return joblib.load(github_url + file_name)

def load_csv_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        csv_data = StringIO(response.content.decode('utf-8'))
        df = pd.read_csv(csv_data)
        return df
    else:
        response.raise_for_status()

def semantic_search(model, search_term, top_n=5):
    try:
        search_term_vector = model.wv[search_term]
    except:
        return None
    similarities = []
    for word in model.wv.index_to_key:
        if word == search_term:
            continue
        word_vector = model.wv[word]
        sim = cosine_similarity([search_term_vector], [word_vector])
        similarities.append((word, sim[0][0]))

    return sorted(similarities, key=lambda item: -item[1])[:top_n]

def answer_question(question, context, qa_pipeline):
    result = qa_pipeline({'question': question, 'context': context})
    return result['answer']

def find_closest_books(input_name, new_tfidf_vectorizer, new_tfidf_matrix,book_df, top_n=3):
    # Vectorize the input using the same TF-IDF vectorizer
    input_vector = new_tfidf_vectorizer.transform([input_name])

    # Compute cosine similarity between the input and all book names
    cosine_similarities = cosine_similarity(input_vector, new_tfidf_matrix).flatten()

    # Get the indices of the top_n closest books
    closest_indices = np.argsort(-cosine_similarities)[:top_n]

    # Get the names of the top_n closest books
    closest_books = book_df['book_name'].iloc[closest_indices].tolist()
    return closest_books

@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")

@st.cache_resource
def load_csv_from_github(url):
    return pd.read_csv(url)

@st.cache_resource
def download_file(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        response.raise_for_status()

def interactive_qa_t5(df, new_tfidf_vectorizer, new_tfidf_matrix, qa_pipeline):
    book_name = st.text_input("Enter a book name to ask about:").strip()

    if book_name:
        if book_name.lower() == 'exit':
            st.stop()

        if book_name not in df['book_name'].values:
            closest_books = find_closest_books(book_name, new_tfidf_vectorizer, new_tfidf_matrix, df, top_n=5)
            st.write("Did you mean one of these books?")
            for name in closest_books:
                st.write(f"- {name}")

            # User can then modify their input based on suggestions
            st.write("Please enter the correct book name in the input box above (or type 'exit' to quit).")
        else:
            summary = df[df['book_name'] == book_name]['cleaned_summary'].iloc[0]
            question = st.text_input("What is your question about the book?")
            if question:
                with st.spinner('Finding the answer...'):
                    answer = answer_question(question, summary, qa_pipeline)
                    st.write(f"Answer: {answer}")

# Main
def main():
    st.title(":books: Welcome to the NLP Project 2 App : Book Analysis and recommandation models :books:")
    st.write("Created by Anna ZENOU and Timothe VITAL")

    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Home", "Sentiment Analysis", "Semantic Search", "Question Answering", "Book recommendation"])
    if app_mode == "Book recommendation":
        tfidf_vectorizer_similar_book = TfidfVectorizer(stop_words='english')
        book_df = load_csv_from_github('https://raw.githubusercontent.com/Timothevtl/NLP_project_app/main/book_df.csv')
        tfidf_matrix_similar_book = tfidf_vectorizer_similar_book.fit_transform(book_df['cleaned_summary'])
        user_query = st.text_input("Enter a query, for example : 'A book about wizards'")
        if st.button("Find similar book"):
            recommended_books = find_similar_books(user_query,tfidf_vectorizer_similar_book, tfidf_matrix_similar_book, book_df['book_name'], book_df['summary_summary'],book_df['average_rating'], 5)
            for book, score, summary, rating in recommended_books:
                with st.expander(f"{book} (Similarity Score: {score}, Rating: {rating})"):
                    st.write(f"Summary: {summary}")
            
    elif app_mode == "Sentiment Analysis":
        label_encoder = LabelEncoder().fit(['negative', 'neutral', 'positive'])
        st.title("Sentiment Analysis of Book Reviews")
        model_choice = st.selectbox("Select a model for analysis", ["RandomForest"])

        
        # URL of the zipped file
        zip_url = "https://github.com/Timothevtl/NLP_repository/raw/main/balanced_tfidf_vectorizer.zip"
        zip_path = "balanced_tfidf_vectorizer.zip"

        # Download and unzip
        download_and_unzip(zip_url, zip_path)

        # Load TF-IDF Vectorizer and Optimized RandomForest Model
        tfidf_vectorizer = joblib.load("balanced_tfidf_vectorizer.joblib")
        model = joblib.load("balanced_optimized_rf_model.joblib")
        review_text = st.text_area("Enter the review text here")
        if st.button("Analyze Sentiment"):
            sentiment, score = predict_sentiment(review_text, model, tfidf_vectorizer, label_encoder)
            st.write("Sentiment:", sentiment)
            st.write("Confidence Score:", score)

    elif app_mode == "Semantic Search":
        st.title("Semantic Search with Word2Vec")
        choice = st.selectbox("Select a word2vec model", ["trained on 1000 book summaries", "trained on 60 000 user reviews"])
        if choice == "trained on 1000 book summaries":
            if st.button('Information about the model'):
                st.write('This model was trained on the top 1000 most commented and best rated book summaries found on goodreads.com')
                st.write('This model is quite unprecise, because it was trained on too little data')
            model_original_url = "https://github.com/Timothevtl/NLP_project_app/raw/main/word2vec_model.model"
            download_file(model_original_url, "word2vec_model.model")
            word2vec_model = Word2Vec.load("word2vec_model.model")
        elif choice == "trained on 60 000 user reviews":
            if st.button('Information about the optimized model'):
                st.write('This model was trained on 60 000 book reviews of users from goodreads.com')
                st.write('This model is much better than the other one')
            # URLs for the model and .npy files
            model_fined_tuned_url = "https://github.com/Timothevtl/NLP_repository/raw/main/word2vec_finetuned.model"
            npy_file1_url = "https://github.com/Timothevtl/NLP_repository/raw/main/word2vec_finetuned.model.syn1neg.npy"
            npy_file2_url = "https://github.com/Timothevtl/NLP_repository/raw/main/word2vec_finetuned.model.wv.vectors.npy"

            # Download the model and .npy files
            download_file(model_fined_tuned_url, "word2vec_finetuned.model")
            download_file(npy_file1_url, "word2vec_finetuned.model.syn1neg.npy")
            download_file(npy_file2_url, "word2vec_finetuned.model.wv.vectors.npy")

            # Load the model from the downloaded files
            word2vec_model = Word2Vec.load("word2vec_finetuned.model")
    
        # UI elements for semantic search
        search_term = st.text_input("Enter a word for semantic search")
        if st.button("Search"):
            similar_words = semantic_search(word2vec_model, search_term, top_n=10)
            if similar_words:
                df = pd.DataFrame(similar_words, columns=["Word", "Similarity Score"])
                st.write('Here are the top 10 closest words to', search_term,'from the user book reviews')
                st.table(df)
            else:
                st.write('The prompted word is not present in the model\'s vocabulary')
                st.write('you might have misspelled? In any case, try typing another word')

    elif app_mode == "Question Answering":
        st.title("Question Answering with BART")

        # Display a message while loading the model
        with st.spinner('Loading Question Answering model...'):
            qa_pipeline = load_qa_pipeline()

        # Load and display the dataset (with caching to avoid re-loading)
        book_df = load_csv_from_github('https://raw.githubusercontent.com/Timothevtl/NLP_project_app/main/book_df.csv')
        new_tfidf_vectorizer = TfidfVectorizer()
        new_tfidf_matrix = new_tfidf_vectorizer.fit_transform(book_df['book_name'])
            
        answer = interactive_qa_t5(book_df,new_tfidf_vectorizer, new_tfidf_matrix, qa_pipeline)
        if answer:
            st.write("Answer:", answer)

        if st.button('Additionnal information'):
            st.write('This functionnality is based on a pre-trained BART Question Answering model that was fined tuned on 1000 book summaries')
            st.write('This model is extractive, which means it looks for an answer in its knowledge, but can\'t generate original content')
            st.write('Don\'t be too precise with your questions, questions that generaly work are for example :')
            st.write('- Where does the story take place?')
            st.write('- What happens in the book?')
            st.write('- Who are the characters?')
    

# Run the app
if __name__ == "__main__":
    main()
