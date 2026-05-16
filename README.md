# VecLens

> A modular recommendation engine built on vector embeddings — mapping users, items, and queries into a shared semantic space to surface personalized results across domains like media, careers, and content.

---

## What it does

VecLens takes a user's text description of their preferences and recommends the closest matching items from a catalog using vector similarity search. Currently built on TF-IDF embeddings with cosine similarity, designed to scale toward sentence-transformers and neural embeddings.

---

## Project structure

```
VecLens/
├── data/
│   ├── index.txt              # list of all catalog item names
│   ├── openness.txt           # description for each personality / item
│   ├── conscientiousness.txt
│   ├── extraversion.txt
│   ├── agreeableness.txt
│   └── neuroticism.txt
│
├── embedding.py               # TFIDFVectorizer class (custom, built from scratch)
├── building_embd.py           # fits vectorizer on catalog, saves matrix + vectorizer
├── VecLens.py                 # user-facing recommendation script
├── visualize.py               # recommendation scores + heatmap visualization
├── plot.py                    # vocabulary explorer + 2D PCA space plots
│
├── tfidf_matrix.npy           # saved embedding matrix
├── vectorizer.pkl             # saved fitted vectorizer (vocab + IDF weights)
├── personalities.pkl          # saved catalog titles
│
└── requirements.txt
```

---

## How it works

```
catalog .txt files
        ↓
TFIDFVectorizer.fit()       builds vocabulary, computes IDF weights
        ↓
TFIDFVectorizer.transform() converts each description into a vector
        ↓
matrix saved to disk        tfidf_matrix.npy + vectorizer.pkl
        ↓
user types a description
        ↓
vectorizer.transform()      maps user input into same vector space
        ↓
cosine similarity           scores every catalog item against user vector
        ↓
ranked recommendations
```

---

## Quickstart

**1. Clone and set up environment**

```bash
git clone https://github.com/Vedansh7-7/VecLens.git
cd VecLens

python -m venv venv
venv\Scripts\activate        # windows
source venv/bin/activate     # mac/linux

pip install -r requirements.txt
```

**2. Build the embeddings**

```bash
python building_embd.py
```

Fits the vectorizer on your catalog and saves `tfidf_matrix.npy`, `vectorizer.pkl`, `personalities.pkl`.

**3. Get recommendations**

```bash
python VecLens.py
```

Enter a description of your preferences when prompted.

**4. Visualize**

```bash
python visualize.py    # recommendation scores, heatmap
python plot.py         # vocabulary explorer, 2D word/title space
```

---

## Adding new items to the catalog

1. Create `data/your_item.txt` with a description
2. Add `your_item` to `data/index.txt`
3. Re-run `python building_embd.py`

No code changes needed.

---

## Tech stack

| Component | Library |
|---|---|
| Vector math | `numpy` |
| Embeddings | custom TF-IDF (`embedding.py`) |
| Visualization | `matplotlib` |
| Dimensionality reduction | `sklearn` PCA |
| Serialization | `pickle` |

---

## Roadmap

- [x] Custom TF-IDF vectorizer from scratch
- [x] Cosine similarity search
- [x] Vocabulary + 2D PCA visualization
- [x] File-based catalog (`.txt` per item)
- [ ] Punctuation stripping + stop word cleanup
- [ ] Weighted mean for user taste vector
- [ ] Sentence-transformers embeddings
- [ ] Multi-domain support (media, careers, content)
- [ ] FastAPI wrapper
- [ ] Web UI

---

## License

MIT