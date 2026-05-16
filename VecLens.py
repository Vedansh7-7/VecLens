import numpy as np
import pickle
from embedding import TFIDFVectorizer, normalised_dot_product

vectorizer = TFIDFVectorizer.load(r".\data\vectorizer.pkl")
matrix     = np.load(r".\data\tfidf_matrix.npy")

with open(r".\data\personalities.pkl", "rb") as f:
    titles = pickle.load(f)

user_input  = input("Describe what your personality: ")
user_vector = vectorizer.transform([user_input])[0]  # same coordinate system

scores = [(titles[i], normalised_dot_product(user_vector, matrix[i])) 
          for i in range(len(titles))]
scores.sort(key=lambda x: x[1], reverse=True)
if scores[0][1] == 0:
    print("\nInput was understood but matched nothing in the catalog.")
    print("Try writing more.")
else:
    print(scores[:1])