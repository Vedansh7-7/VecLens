import numpy as np
import pickle
from embedding import TFIDFVectorizer, normalised_dot_product
import gradio as gr


BOLD_GREEN = "\033[1;32m"
BOLD_YELLOW = "\033[1;33m"
RESET = "\033[0m"

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
    print(f"\nInput was understood but matched nothing in the catalog.")
    print(f"{BOLD_YELLOW}Try writing more.{RESET}")
else:
    print(f"Your character traits are more similar to - {BOLD_GREEN}{scores[:1][0][0]}{RESET}")