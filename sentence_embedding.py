import numpy as np

personality1 = np.array([0.1, 0.2, 0.3, 0.45, 0.5])
personality2 = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
personality3 = np.array([0.10, 0.90, 0.10, 0.60, 0.40])


def dot_product(vec1, vec2):
    return np.dot(vec1, vec2)

print(f" Dot Product of per1 * per2: {dot_product(personality1, personality2):.02f}")
print(f" Dot Product of per1 * per3: {dot_product(personality1, personality3):.02f}")
print(f" Dot Product of per2 * per3: {dot_product(personality2, personality3):.02f}")

# Problem: a longer vector always has a higher dot product even if it's not actually more similar. So we normalize.

# Now using normalisation:

def normalize(vec):
    return vec/np.linalg.norm(vec)

# now using it again

def normalised_dot_product(vec1, vec2):
    return np.dot(normalize(vec1), normalize(vec2))


print(f" Normalised Dot Product of per1 * per2: {normalised_dot_product(personality1, personality2):.02f}")
print(f" Normalised Dot Product of per1 * per3: {normalised_dot_product(personality1, personality3):.02f}")
print(f" Normalised Dot Product of per2 * per3: {normalised_dot_product(personality2, personality3):.02f}")

# Now thinking of Building a vector embedding model, rn using Tf-IDf

# Let traits be a list of descriptions of each of the traits

personalities = ['Neuroticism',
                 'Openness',
                 'Extraversion',
                 'Agreeableness',
                 'Conscientiousness']

Traits = ['Research links high levels of neuroticism with an increased risk of certain mental health issues. If you’re highly neurotic—meaning you tend to experience a lot of negative feelings like fear, depression, and anger—you’re more likely to feel overwhelmed by stressful situations. While another person might take a parking ticket in their stride, for example, you may see it as a catastrophe that ruins your day. You’re also more likely to belittle yourself for minor mistakes that other people simply shrug off.',
          'Approaching the world with an openness to new experiences can be seen as a positive personality trait, unless that openness crosses over into excessive risk-taking. Acquiring knowledge, meeting new people, and trying out new hobbies are also great ways to keep your brain active and maintain healthy cognitive functioning as you age.',
          'Being a social butterfly can come with many benefits. If you’re an extravert, you likely have higher self-esteem, find it easier to adapt to life’s changes, and enjoy a greater overall sense of well-being. Part of this may be because extraverts often have more social support and are more likely to seek help from others.',
          'As with extraversion, people with high agreeableness tend to enjoy a greater sense of social well-being. If you’re agreeable, friends may gravitate toward your generous and trusting personality. Those very friends form a social support network that helps you navigate life’s challenges and better cope with stress.',
          'The more conscientious you are, the more likely you are to take a responsible approach to life. This can have implications for mental and physical health, as well as overall success. You’re more likely to take your physical health seriously, by regularly exercising and seeing your doctor. And you’re likely a diligent employee or student, with an achievement-oriented mindset.',
          ]             # Source: https://www.helpguide.org/mental-health/psychology/personality-types-traits-and-how-it-affects-mental-health

vocab = sorted(set(" ".join(Traits).split()))
print(vocab)
N = len(Traits)

# Defining word frequencies in each document
def tf(word, doc):
    words = doc.split()
    return words.count(word) / len(words)

# Defining the frequnecies of a word in vocab

def idf(word, docs):
    containing_N = 0
    for i in range(N):
        if word in docs[i].split():
            containing_N += 1
    idf_of_word = np.log( N / containing_N)
    return idf_of_word

def tfidf(word, doc, docs):
    return tf(word, doc) * tf(word, docs)

matrix = []

for trait in Traits:
    row = [tfidf(word, trait, Traits) for word in vocab]
    matrix.append(row)


matrix = np.array(matrix)
print(matrix.shape)