from embedding import TFIDFVectorizer
import numpy as np
import pickle

import os

def load_catalog(data_dir="data"):
    # read index
    with open(f"{data_dir}/index.txt", "r") as f:
        names = [line.strip() for line in f.readlines()]
    
    personalities = []
    descriptions = []
    
    for name in names:
        path = f"{data_dir}/{name}.txt"
        with open(path, "r") as f:
            text = f.read().strip()
        
        personalities.append(name.replace("_", " ").title()) 
        descriptions.append(text)
    
    return personalities, descriptions

personalities, descriptions = load_catalog()

vectorizer = TFIDFVectorizer()
matrix = vectorizer.fit_transform(descriptions)

print(vectorizer.get_feature_names())
print(matrix.shape)

np.save(r".\data\tfidf_matrix.npy", matrix)
vectorizer.save(r".\data\vectorizer.pkl")

with open(r".\data\personalities.pkl", "wb") as f:
    pickle.dump(personalities, f)