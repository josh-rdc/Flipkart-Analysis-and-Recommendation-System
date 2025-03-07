from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - {"not", "no"}  # Keep negations
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters and digits but keep words
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lemmatization (optional)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    
    return text

def create_similarity_matrix(df):
    # Preprocess text
    df['description'] = df['description'].apply(preprocess_text)
    
    # Create TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words="english", max_features=300)
    tfidf_matrix = tfidf.fit_transform(df['description'])
    
    cosine_sim = cosine_similarity(tfidf_matrix,
                                tfidf_matrix)

    return cosine_sim

