import numpy as np
import pickle
import streamlit as st
from embedding import TFIDFVectorizer, normalised_dot_product

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VecLens",
    page_icon="🔍",
    layout="centered"
)

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0d0d0d;
    color: #e8e8e8;
}

.stApp {
    background-color: #0d0d0d;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.title-block {
    margin-top: 2rem;
    margin-bottom: 0.2rem;
}

.title-block h1 {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -2px;
    color: #ffffff;
    margin: 0;
    line-height: 1;
}

.title-block p {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #555;
    margin-top: 0.4rem;
    letter-spacing: 0.05em;
}

.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 1.5rem 0;
}

textarea {
    background-color: #151515 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    color: #e8e8e8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
}

textarea:focus {
    border-color: #4fffb0 !important;
    box-shadow: 0 0 0 1px #4fffb0 !important;
}

.stButton > button {
    background-color: #4fffb0;
    color: #0d0d0d;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    border: none;
    border-radius: 4px;
    padding: 0.6rem 1.8rem;
    transition: all 0.2s ease;
    width: 100%;
}

.stButton > button:hover {
    background-color: #ffffff;
    transform: translateY(-1px);
}

.result-card {
    background: #111;
    border: 1px solid #222;
    border-left: 3px solid #4fffb0;
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.5rem;
}

.result-card .label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    color: #555;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.result-card .personality {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #4fffb0;
    letter-spacing: -1px;
}

.score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1a1a1a;
    font-size: 0.82rem;
}

.score-row:last-child {
    border-bottom: none;
}

.score-row .name {
    color: #aaa;
}

.score-row .bar-wrap {
    flex: 1;
    margin: 0 1rem;
    background: #1a1a1a;
    border-radius: 2px;
    height: 4px;
    overflow: hidden;
}

.score-row .bar-fill {
    height: 100%;
    background: #4fffb0;
    border-radius: 2px;
    transition: width 0.6s ease;
}

.score-row .val {
    color: #555;
    min-width: 3.5rem;
    text-align: right;
    font-size: 0.75rem;
}

.warn-box {
    background: #1a1400;
    border: 1px solid #3a3000;
    border-left: 3px solid #f0c040;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    color: #c0a030;
    font-size: 0.82rem;
    margin-top: 1rem;
}

.vocab-hint {
    font-size: 0.72rem;
    color: #444;
    margin-top: 1rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── load data ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    vectorizer = TFIDFVectorizer.load(r".\data\vectorizer.pkl")
    matrix     = np.load(r".\data\tfidf_matrix.npy")
    with open(r".\data\personalities.pkl", "rb") as f:
        titles = pickle.load(f)
    return vectorizer, matrix, titles

vectorizer, matrix, titles = load_data()

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>VecLens</h1>
    <p>vector-space personality mapper · big five model · tfidf embeddings</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── input ─────────────────────────────────────────────────────────────────────
user_input = st.text_area(
    label="",
    placeholder="Describe your personality, habits, and how you approach the world...",
    height=130,
    label_visibility="collapsed"
)

run = st.button("ANALYSE →")

# ── logic ─────────────────────────────────────────────────────────────────────
if run and user_input.strip():
    user_vector = vectorizer.transform([user_input])[0]

    if np.linalg.norm(user_vector) == 0:
        # show which words were unrecognised
        vocab     = set(vectorizer.get_feature_names())
        words     = set(user_input.lower().split())
        unmatched = words - vocab

        st.markdown(f"""
        <div class="warn-box">
            ⚠ No vocabulary match — none of your words exist in the catalog.<br>
            Unrecognised words: <b>{', '.join(sorted(unmatched)) or '—'}</b>
        </div>
        <p class="vocab-hint">Try words like: {', '.join(list(vocab)[:18])}...</p>
        """, unsafe_allow_html=True)

    else:
        scores = [(titles[i], float(normalised_dot_product(user_vector, matrix[i])))
                  for i in range(len(titles))]
        scores.sort(key=lambda x: x[1], reverse=True)

        if scores[0][1] == 0:
            st.markdown("""
            <div class="warn-box">
                ⚠ Input understood but matched nothing — try writing more descriptively.
            </div>
            """, unsafe_allow_html=True)

        else:
            top_name  = scores[0][0]
            top_score = scores[0][1]

            # top result card
            st.markdown(f"""
            <div class="result-card">
                <div class="label">closest personality trait</div>
                <div class="personality">{top_name}</div>
            </div>
            """, unsafe_allow_html=True)

            # all scores breakdown
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.72rem;color:#444;letter-spacing:0.1em;text-transform:uppercase;'>all scores</p>", unsafe_allow_html=True)

            max_score = scores[0][1] if scores[0][1] > 0 else 1
            rows_html = ""
            for name, score in scores:
                pct = (score / max_score) * 100
                rows_html += f"""
                <div class="score-row">
                    <span class="name">{name}</span>
                    <div class="bar-wrap">
                        <div class="bar-fill" style="width:{pct}%"></div>
                    </div>
                    <span class="val">{score:.4f}</span>
                </div>
                """

            st.markdown(rows_html, unsafe_allow_html=True)

elif run and not user_input.strip():
    st.markdown("""
    <div class="warn-box">⚠ Please enter a description first.</div>
    """, unsafe_allow_html=True)