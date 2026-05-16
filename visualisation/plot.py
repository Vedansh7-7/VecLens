import numpy as np
import pickle
import matplotlib.pyplot as plt

# ── load saved data ───────────────────────────────────────────────────────────

matrix = np.load(r"VecLens\tfidf_matrix.npy")

with open(r"VecLens\vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(r"VecLens\personalities.pkl", "rb") as f:
    titles = pickle.load(f)

features = vectorizer.get_feature_names()

# ── user input ────────────────────────────────────────────────────────────────

user_input = input("Describe what you enjoy: ")
user_vector = vectorizer.transform([user_input])[0]

if user_vector.sum() == 0:
    print("No matching words found in vocabulary. Try different words.")
    print("Vocabulary:", features)
    exit()

# ── plot 1: top dims in user taste vector ─────────────────────────────────────

scores = list(zip(features, user_vector))
top = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
top_words, top_scores = zip(*top)

plt.figure(figsize=(10, 5))
plt.barh(top_words, top_scores, color="steelblue")
plt.xlabel("TF-IDF Score")
plt.title("Top dimensions in your taste vector")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("user_taste_vector.png", dpi=150)
plt.show()
print("Saved: user_taste_vector.png")

# ── plot 2: full matrix heatmap ───────────────────────────────────────────────

plt.figure(figsize=(14, 5))
plt.imshow(matrix, aspect="auto", cmap="YlOrRd")
plt.colorbar(label="TF-IDF score")
plt.yticks(range(len(titles)), titles)
plt.xticks(range(len(features)), features, rotation=90, fontsize=7)
plt.title("TF-IDF matrix — all titles vs all dimensions")
plt.tight_layout()
plt.savefig("tfidf_heatmap.png", dpi=150)
plt.show()
print("Saved: tfidf_heatmap.png")

# ── plot 3: similarity scores bar chart ───────────────────────────────────────

def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_scores = [(titles[i], cosine_sim(user_vector, matrix[i]))
              for i in range(len(titles))]
sim_scores.sort(key=lambda x: x[1], reverse=True)

ranked_titles, ranked_scores = zip(*sim_scores)

colors = ["#2ecc71" if s > 0.3 else "#3498db" if s > 0.1 else "#bdc3c7"
          for s in ranked_scores]

plt.figure(figsize=(10, 5))
plt.barh(ranked_titles, ranked_scores, color=colors)
plt.xlabel("Cosine Similarity")
plt.title(f"Recommendations for: '{user_input}'")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("recommendations.png", dpi=150)
plt.show()
print("Saved: recommendations.png")

# ── print ranked results ──────────────────────────────────────────────────────

print("\nRankings:")
for title, score in sim_scores:
    bar = "█" * int(score * 30)
    print(f"  {title:20s}  {score:.3f}  {bar}")