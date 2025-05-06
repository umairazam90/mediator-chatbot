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

# Load dataset
df = pd.read_csv("data/mediators_data.csv")

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Preprocess text
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
df['mediator Biography'] = df['mediator Biography'].fillna('')
df['mediator website'] = df['mediator website'].fillna('')
df['mediator position'].fillna(df['mediator position'].mode()[0], inplace=True)

# Apply preprocessing
df['preprocessed_biography'] = df['mediator Biography'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_biography'])

# Cosine Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend mediators
def recommend_mediators(user_query, df, cosine_sim, top_n=5, filters=None):
    # Preprocess user query
    preprocessed_query = preprocess_text(user_query)
    # Calculate cosine similarity with user query
    query_vector = tfidf_vectorizer.transform([preprocessed_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    # Sort mediators based on cosine similarity
    sorted_indices = np.argsort(cosine_similarities)[::-1]
    # Filter mediators based on user preferences
    if filters:
        filtered_indices = filter_mediators(df, sorted_indices, filters)
    else:
        filtered_indices = sorted_indices
    # Select top N recommended mediators
    top_n_recommendations = filtered_indices[:top_n]
    # Generate recommendation list with additional information
    recommendations = []
    for idx in top_n_recommendations:
        mediator_info = {
            'fullname': df.loc[idx, 'fullname'],
            'email': df.loc[idx, 'mediator email'],
            'website': df.loc[idx, 'mediator website'],
            'biography': df.loc[idx, 'mediator Biography'],
            'reasoning': generate_reasoning(df.loc[idx])  # Generate reasoning based on mediator info
        }
        recommendations.append(mediator_info)
    return recommendations

# Function to generate reasoning for a mediator
def generate_reasoning(mediator_row):
    # Check if 'mediator Biography' contains keywords related to divorce mediation
    biography = mediator_row['mediator Biography'].lower()
    if 'divorce' in biography or 'family law' in biography:
        reasoning = f"{mediator_row['fullname']} specializes in divorce mediation."
    else:
        reasoning = f"{mediator_row['fullname']} has experience in mediation and conflict resolution."
    return reasoning

# Function to filter mediators based on user preferences
def filter_mediators(df, indices, filters):
    filtered_indices = []
    for idx in indices:
        pass_filter = True
        for key, value in filters.items():
            if key in df.columns and df.loc[idx, key] != value:
                pass_filter = False
                break
        if pass_filter:
            filtered_indices.append(idx)
    return filtered_indices

# Streamlit UI
def main():
    st.title("Mediator Recommendation App")
    user_query = st.text_input("Enter your query:")
    multiple_options = st.checkbox("Specify multiple options for recommendations")
    
    filters = {}
    
    if multiple_options:
        country = st.selectbox("Select mediator country:", df['mediator country'].unique())
        if country:
            filters['mediator country'] = country
        
        state = st.selectbox("Select mediator state:", df['mediator State'].unique())
        if state:
            filters['mediator State'] = state
        
        city = st.selectbox("Select mediator city:", df['mediator city'].unique())
        if city:
            filters['mediator city'] = city
    
    if st.button("Recommend"):
        recommendations = recommend_mediators(user_query, df, cosine_sim, top_n=5, filters=filters)
        st.markdown("**Top 5 Recommended Mediators:**")
        for i, recommendation in enumerate(recommendations):
            st.markdown(f"**{i + 1}. {recommendation['fullname']}**")
            st.write(f"**Email:** {recommendation['email']}")
            st.write(f"**Website:** {recommendation['website']}")
            st.write(f"**Biography:**\n{recommendation['biography'] or 'Biography not available'}")
            st.write(f"**Reasoning:** {recommendation['reasoning']}")
            st.write("---")


if __name__ == "__main__":
    main()