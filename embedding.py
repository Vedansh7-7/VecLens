import numpy as np
import pickle

# personality1 = np.array([0.1, 0.2, 0.3, 0.45, 0.5])
# personality2 = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
# personality3 = np.array([0.10, 0.90, 0.10, 0.60, 0.40])


def dot_product(vec1, vec2):
    return np.dot(vec1, vec2)

# print(f" Dot Product of per1 * per2: {dot_product(personality1, personality2):.02f}")
# print(f" Dot Product of per1 * per3: {dot_product(personality1, personality3):.02f}")
# print(f" Dot Product of per2 * per3: {dot_product(personality2, personality3):.02f}")

# Problem: a longer vector always has a higher dot product even if it's not actually more similar. So we normalize.

# Now using normalisation:

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0: 
        return vec  # Returns the original zero vector safely
    return vec / norm

# now using it again

def normalised_dot_product(vec1, vec2):
    return np.dot(normalize(vec1), normalize(vec2))


# print(f" Normalised Dot Product of per1 * per2: {normalised_dot_product(personality1, personality2):.02f}")
# print(f" Normalised Dot Product of per1 * per3: {normalised_dot_product(personality1, personality3):.02f}")
# print(f" Normalised Dot Product of per2 * per3: {normalised_dot_product(personality2, personality3):.02f}")

# Now thinking of Building a vector embedding model, rn using Tf-IDf
STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "as", "by", "with", "this", "that",
    "was", "are", "be", "been", "have", "has", "had", "you", "your",
    "i", "we", "they", "he", "she", "if", "can", "may", "more",
    "also", "all", "out", "from", "well", "very", "often", "many",
    "like", "just", "so", "do", "its", "into", "than", "their",
    "there", "about", "up", "what", "which", "who", "will", "would",
    "could", "should", "not", "no", "any", "some", "our", "my",
    "those", "these", "them", "his", "her", "come", "look", "help",
    "find", "take", "feel", "tend", "seek", "adapt", "form", "new",
    "high", "lot", "better", "overall", "sense", "part", "those",
    "simply", "seen", "see", "seeing", "because", "toward", "people"
}


import numpy as np
import pickle

class TFIDFVectorizer:
    def __init__(self, stop_words=STOP_WORDS):
        self.vocabulary  = {}
        self.idf_values  = {}
        self.fitted      = False
        self.stop_words  = stop_words   # ← store it

    def fit(self, documents):
        all_words = set()
        for doc in documents:
            for word in doc.split():
                word = word.lower()                        # ← lowercase first
                if word not in self.stop_words:
                    all_words.add(word)

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}

        N = len(documents)
        for word in self.vocabulary:
            containing = sum(1 for doc in documents if word in doc.lower().split())  # ← lowercase doc too
            
            if containing == 0:                            # ← safety net
                self.idf_values[word] = 0.0
            else:
                self.idf_values[word] = np.log(N / containing)

        self.fitted = True
        return self

    def transform(self, documents):
        if not self.fitted:
            raise Exception("Fit the vectorizer first before transforming.")
        
        matrix = []
        for doc in documents:
            words = doc.split()
            row = np.zeros(len(self.vocabulary))
            
            for word in words:
                word = word.lower()                                # ← lowercase before lookup
                if word not in self.vocabulary:
                    continue          # unknown word ko ignore, dimension doesn't exist
                
                tf  = words.count(word) / len(words)
                idf = self.idf_values[word]
                dim = self.vocabulary[word]
                row[dim] = tf * idf
            
            matrix.append(row)
        
        return np.array(matrix)

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self):
        # return words ordered by their dimension index
        return sorted(self.vocabulary, key=self.vocabulary.get)

    def save(self, path=r".\data\vectorizer.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved to {path}")

    @staticmethod
    def load(path=r".\data\vectorizer.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
        
