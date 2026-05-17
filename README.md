# VecLens

> A modular recommendation engine built on vector embeddings — mapping users, items, and queries into a shared semantic space to surface personalized results across domains like media, careers, and content.

---

## What it does

VecLens takes a user's free-text description of their personality and maps it into a shared vector space alongside the Big Five personality traits — then ranks them by cosine similarity.

Built entirely from scratch using only numpy (no sklearn for the core, no pre-trained models) to understand the mathematics of vector embeddings before abstracting them away. Served through a Streamlit web app with three tabs: recommendation results, per-token TF-IDF weight breakdown, and live PCA projections of both the catalog and vocabulary space with the user's position plotted in real time.

---

## Project structure

```
VecLens/
├── data/
│   ├── index.txt              # list of all catalog item names
│   ├── Agreeableness.txt      # description for each personality
│   ├── Conscientiousness.txt
│   ├── Extraversion.txt
│   ├── Neuroticism.txt
│   ├── Openness.txt
│   ├── personalities.pkl      # saved catalog titles
│   ├── tfidf_matrix.npy       # saved embedding matrix
│   └── vectorizer.pkl         # saved fitted vectorizer (vocab + IDF weights)
│
├── visualisation/
│   ├── titles_2d_space.png    # PCA projection of catalog items
│   ├── vocab_2d_space.png     # PCA projection of vocabulary words
│   └── vocab_idf.png          # word rarity bar chart
│
├── embedding.py               # TFIDFVectorizer class (custom, numpy only)
├── building_embd.py           # fits vectorizer on catalog, saves matrix + vectorizer
├── app.py                     # Streamlit web app (3 tabs)
├── VecLens.py                 # CLI recommendation script
├── plot.py                    # vocabulary explorer + 2D PCA space plots
├── README.md
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
ranked recommendations + visualizations
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

Fits the vectorizer on your catalog and saves `tfidf_matrix.npy`, `vectorizer.pkl`, `personalities.pkl` inside `data/`.

**3. Run the web app**

```bash
streamlit run app.py
```

**4. Or use the CLI**

```bash
python VecLens.py
```

**5. Standalone plots**

```bash
python plot.py     # vocabulary explorer, 2D PCA space
```

---

## App tabs

| Tab | What it shows |
|---|---|
| **Result** | Top matched personality + all scores as bars |
| **Token Weights** | TF-IDF weight of each matched token, IDF rarity chart, unrecognised words |
| **PCA Space** | Titles in 2D with your position as a star · Vocabulary scatter with your tokens highlighted in green |

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
| Web app | `streamlit` |
| Visualization | `matplotlib` |
| Dimensionality reduction | `sklearn` PCA |
| Serialization | `pickle` |

---

## Roadmap

- [x] Custom TF-IDF vectorizer from scratch
- [x] Cosine similarity search
- [x] Vocabulary + 2D PCA visualization
- [x] File-based catalog (`.txt` per item)
- [x] Streamlit web app with token weight + PCA tabs
- [ ] Punctuation stripping + stop word cleanup
- [ ] Weighted mean for user taste vector (recency, ratings)
- [ ] Sentence-transformers embeddings
- [ ] Multi-domain support (media, careers, content)
- [ ] FastAPI wrapper
- [ ] Knowledge graph layer for explainability

---

## License

MIT