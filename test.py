import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources (comment out if already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load mediator dataset
mediator_df = pd.read_csv("/home/test/mediator/data/mediators_data.csv")
# Remove unnamed columns
mediator_df = mediator_df.loc[:, ~mediator_df.columns.str.contains('^Unnamed')]
# Preprocess mediator text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    text = ' '.join(tokens)
    return text

# Handle missing values
mediator_df['mediator Biography'] = mediator_df['mediator Biography'].fillna('')
mediator_df['mediator website'] = mediator_df['mediator website'].fillna('')
mediator_df['mediator position'].fillna(mediator_df['mediator position'].mode()[0], inplace=True)

# Apply preprocessing
mediator_df['preprocessed_biography'] = mediator_df['mediator Biography'].apply(preprocess_text)

# TF-IDF Vectorization for mediators
tfidf_vectorizer_mediator = TfidfVectorizer()
tfidf_matrix_mediator = tfidf_vectorizer_mediator.fit_transform(mediator_df['preprocessed_biography'])

# Cosine Similarity Matrix for mediators
cosine_sim_mediator = cosine_similarity(tfidf_matrix_mediator, tfidf_matrix_mediator)

# Load case description dataset
case_df = pd.read_csv("/home/test/mediator/data/CaseClassifyData.csv")

# Preprocess case description text
case_df['processed_description'] = case_df['caseDescription'].apply(preprocess_text)

# TF-IDF Vectorization for case descriptions
tfidf_vectorizer_case = TfidfVectorizer()
tfidf_matrix_case = tfidf_vectorizer_case.fit_transform(case_df['processed_description'])

# Function to recommend mediators based on the case description
def recommend_mediators(case_description, df, tfidf_vectorizer, tfidf_matrix):
    # Preprocess case description
    processed_description = preprocess_text(case_description)
    # Vectorize case description
    description_vector = tfidf_vectorizer.transform([processed_description])
    # Calculate cosine similarity between case description and mediator biographies
    similarity_scores = cosine_similarity(description_vector, tfidf_matrix)
    # Find top matching mediators
    top_matches = similarity_scores.argsort()[0][::-1][:5]  # Get indices of top 5 matches
    recommended_mediators = df.iloc[top_matches]
    return recommended_mediators

# Streamlit UI
def main():
    st.title("Mediator Recommendation App")
    case_description = st.text_area("Describe your case:")
    
    if st.button("Recommend Mediators"):
        recommended_mediators = recommend_mediators(case_description, mediator_df, tfidf_vectorizer_mediator, tfidf_matrix_mediator)
        st.markdown("**Top 5 Recommended Mediators:**")
        for idx, mediator_info in recommended_mediators.iterrows():
            st.markdown(f"**{mediator_info['fullname']}**")
            st.write(f"**Email:** {mediator_info['mediator email']}")
            st.write(f"**Website:** {mediator_info['mediator website']}")
            st.write(f"**Biography:**\n{mediator_info['mediator Biography']}")
            st.write("---")

if __name__ == "__main__":
    main()
