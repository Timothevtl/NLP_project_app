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

nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    # Remove stopwords
    filtered_words = [word for word in words if word not in english_stopwords]
    return " ".join(filtered_words)

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
    return joblib.load(file_name)

def semantic_search(model, search_term, top_n=5):
    search_term_vector = model.wv[search_term]
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

def interactive_qa_t5(df, new_tfidf_matrix, qa_pipeline):
    while True:
        book_name = input("Enter a book name to ask about (or 'exit' to quit): ").strip()
        if book_name.lower() == 'exit':
            break

        if book_name not in df['book_name'].values:
            closest_books = find_closest_books(book_name, new_tfidf_matrix, df, top_n=5)
            print("Did you mean one of these books?")
            for name in closest_books:
                print(f"- {name}")
            book_name = input("Please enter the correct book name (or 'exit' to quit): ").strip()
            if book_name.lower() == 'exit':
                break

        if book_name in df['book_name'].values:
            summary = df[df['book_name'] == book_name]['cleaned_summary'].iloc[0]
            question = input("What is your question about the book? ")
            answer = answer_question(question, summary, qa_pipeline)
            print(f"Answer: {answer}\n")
        else:
            print("Book not found. Please try again.\n")

# Main
def main():
    st.title("Book Analysis Application")
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Home", "Sentiment Analysis", "Semantic Search", "Question Answering"])

    if app_mode == "Sentiment Analysis":
        label_encoder = LabelEncoder().fit(['negative', 'neutral', 'positive'])
        st.title("Sentiment Analysis of Book Reviews")
        model_choice = st.selectbox("Select a model for analysis", ["XGBoost", "RandomForest"])

        # Define GitHub URL
        github_url = "https://raw.githubusercontent.com/Timothevtl/NLP_project_app/main/"

        # Load TF-IDF Vectorizer
        tfidf_vectorizer = load_model_from_github('tfidf_vectorizer.joblib', github_url)

        # Load the chosen model
        if model_choice == "XGBoost":
            model = xgb.Booster()
            model.load_model('https://raw.githubusercontent.com/Timothevtl/NLP_project_app/main/xgboost_model.json')
        else:
            model = load_model_from_github('saved_randomforest_model.joblib', github_url)
        review_text = st.text_area("Enter the review text here")
        if st.button("Analyze Sentiment"):
            sentiment, score = predict_sentiment(review_text, model, tfidf_vectorizer, label_encoder)
            st.write("Sentiment:", sentiment)
            st.write("Confidence Score:", score)

    elif app_mode == "Semantic Search":
        st.title("Semantic Search with Word2Vec")
        # Load Word2Vec model only if this option is chosen
        word2vec_model = Word2Vec.load("https://raw.githubusercontent.com/Timothevtl/NLP_project_app/main/word2vec_model.model")
    
        # UI elements for semantic search
        search_term = st.text_input("Enter a word for semantic search")
        if st.button("Search"):
            similar_words = semantic_search(word2vec_model, search_term, top_n=10)
            st.write("Similar words:", similar_words)

    elif app_mode == "Question Answering":
        st.title("Question Answering with BART")
        st.write('Loading models...')
        # Load BART QA model only if this option is chosen
        qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")
        book_df = pd.read_csv('https://raw.githubusercontent.com/Timothevtl/NLP_project_app/main/1000_best_books_summary.csv')
        new_tfidf_vectorizer = load_model_from_github('QA_tfidf_vectorizer.joblib', github_url)
        new_tfidf_matrix = new_tfidf_vectorizer.fit_transform(book_df['book_name'])
        if st.button("Get Answer"):
            answer = interactive_qa_t5(book_df, new_tfidf_matrix, qa_pipeline)
            st.write("Answer:", answer)

# Run the app
if __name__ == "__main__":
    main()