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

# Let traits be a list of descriptions of each of the traits

personalities = ['Neuroticism',
                 'Openness',
                 'Extraversion',
                 'Agreeableness',
                 'Conscientiousness']

Traits = ['neuroticism tend to experience a lot of negative feelings like fear, depression, and ange you’re more likely to feel overwhelmed by stressful situations. You’re also more likely to belittle yourself for minor mistakes that other people simply shrug off.',
          'Approaching the world with an openness to new experiences can be seen as a positive personality trait acquiring knowledge, meeting new people, and trying out new hobbies are all people with high openness are more curious and look for novel experiences.',
          'Being a social butterfly can come with many benefits. If you’re an extravert, you likely have self-esteem, find it easier to adapt to life’s changes, and enjoy a greater overall sense of well-being. Part of this may be because extraverts often have more social support and are more likely to seek help from others. people are outgoing, assertive, and expressive.',
          'people with high agreeableness tend to enjoy a greater sense of social well-being. If you’re agreeable, friends may gravitate toward your generous and trusting personality. Those very friends form a social support network that helps you navigate life’s challenges and better cope with stress. are highly agreeable are altruistic, trusting, and cooperative.',
          'The conscientious take a responsible approach to life. This can have implications for mental and physical health, as well as overall success. You’re more likely to take your physical health seriously, by regularly exercising and seeing your doctor. And you’re likely a diligent employee or student, with an achievement-oriented mindset. people are more organized, self-controlled, and focused on goals.',
          ]          # Source: https://www.helpguide.org/mental-health/psychology/personality-types-traits-and-how-it-affects-mental-health

# vocab = sorted(set(" ".join(Traits).split()))
# # print(vocab)
# N = len(Traits)

# # Defining word frequencies in each document
# def tf(word, doc):
#     words = doc.split()
#     return words.count(word) / len(words)

# # Defining the frequnecies of a word in vocab

# def idf(word, docs):
#     containing_N = 0
#     for i in range(N):
#         if word in docs[i].split():
#             containing_N += 1
#     idf_of_word = np.log( N / containing_N)
#     return idf_of_word

# def tfidf(word, doc, docs):
#     return tf(word, doc) * idf(word, docs)

# matrix = []

# for trait in Traits:
#     # print(trait)
#     row = [tfidf(word, trait, Traits) for word in vocab]
#     matrix.append(row)


# matrix = np.array(matrix)
# # print(matrix.shape)

# # print(matrix)


import numpy as np
import pickle

class TFIDFVectorizer:
    def __init__(self):
        self.vocabulary = {}
        self.idf_values = {}
        self.fitted = False

    def fit(self, documents):
        
        all_words = set()
        for doc in documents:
            for word in doc.split():
                all_words.add(word)
        
        # assign each word a fixed dimension index
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        
        # compute IDF for each word......
        N = len(documents)
        for word in self.vocabulary:
            containing = sum(1 for doc in documents if word in doc.split())
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

    def save(self, path="vectorizer.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved to {path}")

    @staticmethod
    def load(path="vectorizer.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
        
