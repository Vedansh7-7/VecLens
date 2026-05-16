from embedding import TFIDFVectorizer
import numpy as np
import pickle

# Let us bring here, the traits: a list of descriptions of each of the traits

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
          ]             # Source: https://www.helpguide.org/mental-health/psychology/personality-types-traits-and-how-it-affects-mental-health

vectorizer = TFIDFVectorizer()
matrix = vectorizer.fit_transform(Traits)

print(vectorizer.get_feature_names())
print(matrix.shape)

np.save("tfidf_matrix.npy", matrix)
vectorizer.save("vectorizer.pkl")

with open("personalities.pkl", "wb") as f:
    pickle.dump(personalities, f)