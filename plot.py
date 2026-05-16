# import numpy as np
# import pickle
# import matplotlib.pyplot as plt

# # ── load saved data ───────────────────────────────────────────────────────────

# matrix = np.load(r"D:\Coding2\MLAI\VecLens\VecLens\tfidf_matrix.npy")

# with open(r"D:\Coding2\MLAI\VecLens\VecLens\vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# with open(r"D:\Coding2\MLAI\VecLens\VecLens\personalities.pkl", "rb") as f:
#     titles = pickle.load(f)

# features = vectorizer.get_feature_names()

# # ── user input ────────────────────────────────────────────────────────────────

# user_input = input("Describe what you enjoy: ")
# user_vector = vectorizer.transform([user_input])[0]

# if user_vector.sum() == 0:
#     print("No matching words found in vocabulary. Try different words.")
#     print("Vocabulary:", features)
#     exit()

# # ── plot 1: top dims in user taste vector ─────────────────────────────────────

# scores = list(zip(features, user_vector))
# top = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
# top_words, top_scores = zip(*top)

# plt.figure(figsize=(10, 5))
# plt.barh(top_words, top_scores, color="steelblue")
# plt.xlabel("TF-IDF Score")
# plt.title("Top dimensions in your taste vector")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig("user_taste_vector.png", dpi=150)
# plt.show()
# print("Saved: user_taste_vector.png")

# # ── plot 2: full matrix heatmap ───────────────────────────────────────────────

# plt.figure(figsize=(14, 5))
# plt.imshow(matrix, aspect="auto", cmap="YlOrRd")
# plt.colorbar(label="TF-IDF score")
# plt.yticks(range(len(titles)), titles)
# plt.xticks(range(len(features)), features, rotation=90, fontsize=7)
# plt.title("TF-IDF matrix — all titles vs all dimensions")
# plt.tight_layout()
# plt.savefig("tfidf_heatmap.png", dpi=150)
# plt.show()
# print("Saved: tfidf_heatmap.png")

# # ── plot 3: similarity scores bar chart ───────────────────────────────────────

# def cosine_sim(a, b):
#     if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
#         return 0.0
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# sim_scores = [(titles[i], cosine_sim(user_vector, matrix[i]))
#               for i in range(len(titles))]
# sim_scores.sort(key=lambda x: x[1], reverse=True)

# ranked_titles, ranked_scores = zip(*sim_scores)

# colors = ["#2ecc71" if s > 0.3 else "#3498db" if s > 0.1 else "#bdc3c7"
#           for s in ranked_scores]

# plt.figure(figsize=(10, 5))
# plt.barh(ranked_titles, ranked_scores, color=colors)
# plt.xlabel("Cosine Similarity")
# plt.title(f"Recommendations for: '{user_input}'")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig("recommendations.png", dpi=150)
# plt.show()
# print("Saved: recommendations.png")

# # ── print ranked results ──────────────────────────────────────────────────────

# print("\nRankings:")
# for title, score in sim_scores:
#     bar = "█" * int(score * 30)
#     print(f"  {title:20s}  {score:.3f}  {bar}")

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

# ── load ──────────────────────────────────────────────────────────────────────

matrix = np.load(r"D:\Coding2\MLAI\VecLens\VecLens\data\tfidf_matrix.npy")

with open(r"D:\Coding2\MLAI\VecLens\VecLens\data\vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(r"D:\Coding2\MLAI\VecLens\VecLens\data\personalities.pkl", "rb") as f:
    titles = pickle.load(f)

features = np.array(vectorizer.get_feature_names())

# ── plot 1: vocabulary list with global IDF weights ───────────────────────────
# higher IDF = rarer word = more discriminating

idf_scores = np.array([vectorizer.idf_values[w] for w in features])
sorted_idx = np.argsort(idf_scores)[::-1]   # descending

print("=" * 45)
print(f"{'WORD':<20} {'IDF SCORE':>10}  {'RARITY'}")
print("=" * 45)
for i in sorted_idx:
    bar = "█" * int(idf_scores[i] * 8)
    print(f"{features[i]:<20} {idf_scores[i]:>10.3f}  {bar}")
print("=" * 45)
print(f"Total vocab size: {len(features)} words")

# ── plot 2: IDF scores bar chart ──────────────────────────────────────────────

plt.figure(figsize=(12, 5))
plt.bar(features[sorted_idx], idf_scores[sorted_idx], color="steelblue")
plt.xticks(rotation=90, fontsize=8)
plt.ylabel("IDF Score (higher = rarer)")
plt.title("Vocabulary — word rarity across your catalog")
plt.tight_layout()
plt.savefig(r".\visualisation\vocab_idf.png", dpi=150)
plt.show()
print("Saved: vocab_idf.png")

# ── plot 3: words in 2D space via PCA ────────────────────────────────────────
# each word is a column in the matrix (how it scores across all documents)
# we transpose so each word becomes a row vector, then project to 2D

word_vectors = matrix.T   # shape: (n_words, n_titles)

if word_vectors.shape[0] < 2 or word_vectors.shape[1] < 2:
    print("Not enough data to run PCA — add more titles/words.")
else:
    n_components = min(2, word_vectors.shape[0], word_vectors.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(word_vectors)   # shape: (n_words, 2)

    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(13, 8))

    # color words by IDF — rarer words get warmer color
    colors = cm.coolwarm(idf_scores / idf_scores.max())

    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=60, alpha=0.8, zorder=3)

    for i, word in enumerate(features):
        ax.annotate(
            word,
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.85
        )

    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.set_title("Vocabulary in 2D space (PCA projection)\nwarm = rare word, cool = common word")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(r".\visualisation\vocab_2d_space.png", dpi=150)
    plt.show()
    print("Saved: vocab_2d_space.png")

# ── plot 4: titles in 2D space ────────────────────────────────────────────────
# same idea but for documents — each title projected to 2D

if matrix.shape[0] >= 2 and matrix.shape[1] >= 2:
    n_components = min(2, matrix.shape[0], matrix.shape[1])
    pca_docs = PCA(n_components=n_components)
    doc_coords = pca_docs.fit_transform(matrix)

    explained_docs = pca_docs.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 7))

    colors_docs = cm.tab10(np.linspace(0, 1, len(titles)))
    ax.scatter(doc_coords[:, 0], doc_coords[:, 1],
               c=colors_docs, s=120, zorder=3)

    for i, title in enumerate(titles):
        ax.annotate(
            title,
            xy=(doc_coords[i, 0], doc_coords[i, 1]),
            xytext=(8, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold"
        )

    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_xlabel(f"PC1 ({explained_docs[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_docs[1]:.1f}% variance)")
    ax.set_title("Titles in 2D space (PCA projection)\ncloser = more similar vocabulary")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(r".\visualisation\titles_2d_space.png", dpi=150)
    plt.show()
    print("Saved: titles_2d_space.png")