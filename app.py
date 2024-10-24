from flask import Flask, render_template, request, jsonify
import Levenshtein as lev
import spacy
import nltk
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer, util
app = Flask(__name__)

# Load spaCy model and download stopwords
nlp = spacy.load("en_core_web_md")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load FAQs from external JSON file
def load_faq():
    with open(r'C:\Users\LOQ\Documents\code\casual\P_mitra\faq.json', 'r') as f:
        faq_data = json.load(f)
    return faq_data

faq = load_faq()
#print("FAQs loaded:", faq)
# Function to preprocess text (remove stopwords, punctuation, lemmatize, etc.)
def preprocess(text):
    print(f"Original text: {text}")
    tokens = text.lower().split()
    stop_words = set(stopwords.words('english'))
    cleaned_text = ' '.join([word for word in tokens if word not in stop_words])
    print(f"Cleaned text: {cleaned_text}")
    return cleaned_text

 
# Function to get the best matching answer using Cosine Similarity and TF-IDF
# Function to get the best matching answer using SpaCy embeddings

# Load pre-trained BERT model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get the best matching answer using BERT embeddings
def get_best_answer(user_input):
    faq_questions = list(faq.keys())

    
    # Embed both the FAQ questions and the user's input
    faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)
    
    # Compute cosine similarity between user input and FAQ questions
    similarity_scores = util.pytorch_cos_sim(user_input_embedding, faq_embeddings)
    
    # Get the best match
    best_match_index = similarity_scores.argmax().item()
    best_score = similarity_scores[0][best_match_index].item()
    
    # Set a similarity threshold
    threshold = 0.40  # Adjust based on experimentation

    if best_score > threshold:
        return faq[faq_questions[best_match_index]]
    else:
        return "I'm sorry, I don't have the answer to that. Please contact the college administration for more information."

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# API route to handle user queries
@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        # Print to confirm that the request has been received
        print("Request received!")

        # Ensure you're receiving the JSON payload correctly
        user_message = request.json.get('message', '')
        print(f"Received message: {user_message}")

        if not user_message:
            return jsonify({'response': "Please provide a valid question."})

        # Call your function and print the result for debugging
        response = get_best_answer(user_message)
        print(f"Generated response: {response}")
        
        # Return the response
        return jsonify({'response': response})
    
    except Exception as e:
        # Catch and display errors
        print(f"Error occurred: {e}")
        return jsonify({'response': f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
