import numpy as np
import pickle
from embedding import TFIDFVectorizer, normalised_dot_product

vectorizer = TFIDFVectorizer.load("vectorizer.pkl")
matrix     = np.load("tfidf_matrix.npy")

with open("personalities.pkl", "rb") as f:
    titles = pickle.load(f)

user_input  = input("Describe what your personality: ")
user_vector = vectorizer.transform([user_input])[0]  # same coordinate system

scores = [(titles[i], normalised_dot_product(user_vector, matrix[i])) 
          for i in range(len(titles))]

print(max(scores, key=lambda x: x[1]))