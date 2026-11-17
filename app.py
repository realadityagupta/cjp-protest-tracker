"""CJP Protest Tracker dashboard.

Run:  streamlit run app.py
Reads data/final_dataset.csv and data/*.csv produced by CJP_v2.ipynb.
"""
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CJP Protest Tracker", page_icon="📰", layout="wide")
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # data/, models/, charts/ are next to this file

COLORS = {"Protest": "#d1495b", "Leak & Investigation": "#edae49",
          "Court & Govt response": "#00798c", "Not relevant": "#9e9e9e"}


@st.cache_data
def load():
    df = pd.read_csv("data/final_dataset.csv", parse_dates=["Published Date"])
    before_after = pd.read_csv("data/before_after.csv", index_col=0)
    results = pd.read_csv("data/model_vs_rules.csv", index_col=0)
    return df, before_after, results


@st.cache_resource(show_spinner="Loading the AI model (first time only, about 10 seconds)…")
def load_models():
    """Both models saved by the notebook.

    - final model: sentence embeddings (meaning) + Logistic Regression -> used for the dataset and main prediction
    - word model:  TF-IDF (words) + Logistic Regression -> used to explain which words matter
    """
    word_model = joblib.load("models/event_type.joblib")
    try:
        from sentence_transformers import SentenceTransformer
        saved = joblib.load("models/event_type_embeddings.joblib")
        return SentenceTransformer(saved["encoder"]), saved["classifier"], word_model
    except Exception:   # e.g. the AI model can't be downloaded on the server: fall back to the word model only
        return None, None, word_model


df, before_after, results = load()   # the models load only when the 🤖 tab is used, so the dashboard opens fast
FINAL = "Final model (embeddings + LR)"


def ranked_bar(counts, color, label):
    """Horizontal bar chart sorted biggest-first."""
    data = counts.rename_axis(label).reset_index(name="Articles")
    return alt.Chart(data).mark_bar(color=color).encode(
        x=alt.X("Articles:Q"),
        y=alt.Y(f"{label}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        tooltip=[label, "Articles"],
    ).properties(height=alt.Step(24))   # fixed height per bar, so every label is shown


def probability_chart(classes, probs):
    prob_df = pd.DataFrame({"Event Type": classes, "Probability": probs})
    return alt.Chart(prob_df).mark_bar().encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("Event Type:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("Event Type:N", legend=None, scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values()))),
        tooltip=["Event Type", alt.Tooltip("Probability:Q", format=".0%")],
    ).properties(height=170)

st.title("📰 CJP Protest Tracker")
st.caption("Where, why and how did students protest exam paper leaks in India, and how did authorities respond?  "
           f"· {len(df):,} news articles · Event Type predicted by a sentence-embedding model (all-MiniLM-L6-v2 + Logistic Regression)")

# ---------------- Filters ----------------
with st.sidebar:
    st.header("Filters")
    events = st.multiselect("Event Type", list(COLORS), default=[e for e in COLORS if e != "Not relevant"])
    states = st.multiselect("State", sorted(s for s in df["State"].unique() if s != "Unknown"),
                            placeholder="All states")
    exams = st.multiselect("Exam", sorted(e for e in df["Name of Exam"].unique() if e != "Unknown"),
                           placeholder="All exams")
    dated = df["Published Date"].dropna()
    start, end = st.date_input("Published between", (pd.Timestamp("2026-04-01").date(), dated.max().date()),
                               min_value=dated.min().date(), max_value=dated.max().date())
    search = st.text_input("Search headlines")

view = df[df["Event Type"].isin(events)]
if states:
    view = view[view["State"].isin(states)]
if exams:
    view = view[view["Name of Exam"].isin(exams)]
view = view[view["Published Date"].dt.date.between(start, end)]
if search:
    view = view[view["Headline"].str.contains(search, case=False, na=False)]

# ---------------- Headline numbers ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Articles shown", f"{len(view):,}")
c2.metric("Protest articles", f"{(view['Event Type'] == 'Protest').sum():,}")
c3.metric("States with a known state", view.loc[view["State"] != "Unknown", "State"].nunique())
c4.metric("Final model (macro F1, test set)",
          f"{results.loc['Macro F1 (test set)', FINAL]:.2f}",
          f"{results.loc['Macro F1 (test set)', FINAL] - results.loc['Macro F1 (test set)', 'Old keyword rules']:+.2f} vs old rules")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Timeline", "🗺️ States & exams", "🏛️ Organisations", "🔎 Articles",
                                        "🤖 Try the model"])

with tab1:
    weekly = (view.dropna(subset=["Published Date"])
              .groupby([pd.Grouper(key="Published Date", freq="W-MON", label="left", closed="left"), "Event Type"])
              .size().reset_index(name="Articles"))
    st.subheader("Articles per week by Event Type")
    chart = alt.Chart(weekly).mark_bar().encode(
        x=alt.X("yearmonthdate(Published Date):O", title="Week starting (Monday)"),
        y=alt.Y("Articles:Q", scale=alt.Scale(domainMin=0), title="Articles"),   # counts never go below 0
        color=alt.Color("Event Type:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values()))),
        tooltip=["yearmonthdate(Published Date)", "Event Type", "Articles"],
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Google News returns its most recent results first, so the weeks just before collection "
               "(early Aug 2026) are over-represented. Compare the mix of colours, not only bar height.")

with tab2:
    left, right = st.columns(2)
    with left:
        st.subheader("Articles by state")
        st.altair_chart(ranked_bar(view.loc[view["State"] != "Unknown", "State"].value_counts().head(12),
                                   "#d1495b", "State"), width="stretch")
    with right:
        st.subheader("Articles by exam")
        st.altair_chart(ranked_bar(view.loc[view["Name of Exam"] != "Unknown", "Name of Exam"].value_counts().head(12),
                                   "#edae49", "Exam"), width="stretch")

with tab3:
    st.subheader("Organisations mentioned")
    orgs = view["Organisations"].fillna("").str.split(", ").explode()
    st.altair_chart(ranked_bar(orgs[orgs != ""].value_counts().head(15), "#00798c", "Organisation"),
                    width="stretch")

with tab4:
    st.subheader("Articles")
    st.dataframe(
        view[["Published Date", "Headline", "Outlet", "Event Type", "Confidence", "State", "Name of Exam",
              "Organisations", "Link"]].sort_values("Published Date", ascending=False),
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="open"),
                       "Published Date": st.column_config.DatetimeColumn(format="YYYY-MM-DD")},
        hide_index=True, width="stretch",
    )

with tab5:
    st.subheader("Try the model on any headline")
    st.caption("Two models trained on the same 150 hand-labelled headlines. The final model reads the *meaning* "
               "(sentence embeddings); the word model shows *which words* mattered. Each gives every Event Type a "
               "probability and picks the highest.")

    EXAMPLES = [
        "(type your own below)",
        "CJP protesters clash with police during march to Parliament",
        "NEET paper leak: probe finds question paper was sold for Rs 30 lakh",
        "High Court grants bail to accused in recruitment paper leak case",
        "CUET UG 2026 admit card released, check steps to download",
        "Government cancels recruitment exam after question paper leak, new date soon",
        "Hunger strike by CJP students enters 10th day at Jantar Mantar   [a known weak spot]",
    ]
    example = st.selectbox("Pick an example", EXAMPLES)
    headline = st.text_area("Headline", value="" if example == EXAMPLES[0] else example.split("   [")[0], key=f"headline_{example}",
                            placeholder="e.g. Police lathi-charge students protesting paper leak in Ranchi")

    if headline.strip():
        encoder, emb_clf, model = load_models()
        label = None

        # 1) Final model: meaning (sentence embeddings)
        if encoder is None:
            st.info("The meaning model couldn't be loaded here, so only the word model is shown below. "
                    "It runs fully in the notebook and the local dashboard.")
        else:
            emb_probs = emb_clf.predict_proba(encoder.encode([headline], normalize_embeddings=True))[0]
            emb_best = int(emb_probs.argmax())
            label, confidence = emb_clf.classes_[emb_best], emb_probs[emb_best]

            st.markdown("##### 🧠 Final model: reads the *meaning*")
            left, right = st.columns([1, 1.4])
            with left:
                st.markdown(f"#### <span style='color:{COLORS[label]}'>{label}</span>", unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence:.0%}")
                if confidence < 0.5:
                    st.warning("The model is unsure: no label got more than 50%. A human should check this one.")
            with right:
                st.altair_chart(probability_chart(emb_clf.classes_, emb_probs), width="stretch")

        # 2) Word model: counts words, but can show WHICH words mattered
        tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
        probs = model.predict_proba([headline])[0]
        best = int(probs.argmax())
        word_label = model.classes_[best]
        st.markdown("##### 🔤 Second opinion: word model (TF-IDF), which shows *which words* mattered")
        agree = "" if label is None else (": ✅ agrees" if word_label == label else ": ⚠️ disagrees") + " with the final model"
        st.markdown(f"Word model says <span style='color:{COLORS[word_label]}'><b>{word_label}</b></span> "
                    f"({probs[best]:.0%}){agree}.", unsafe_allow_html=True)

        # Why? Each word's push towards the word model's label = its TF-IDF weight x the model's coefficient
        vector = tfidf.transform([headline])
        word_ids = vector.nonzero()[1]
        if len(word_ids) == 0:
            st.info("None of these words appeared in the 150 training headlines, so the word model is only guessing. "
                    "The meaning model can still understand the headline.")
        else:
            pushes = vector.data * clf.coef_[best, word_ids]
            vocab = tfidf.get_feature_names_out()
            why = (pd.DataFrame({"Word or phrase": vocab[word_ids], "Push towards this label": pushes})
                   .sort_values("Push towards this label", ascending=False).head(6))
            st.markdown(f"**Why {word_label}?** The words that pushed the word model towards it "
                        "(negative = pushed away from it):")
            st.dataframe(why.round(2), hide_index=True, width="stretch")

with st.expander("How good is the data? (before vs after, model vs rules)"):
    left, right = st.columns(2)
    left.markdown("**Data quality: before vs after**")
    left.dataframe(before_after.astype(str))
    right.markdown("**Event Type: old keyword rules vs ML model**")
    right.dataframe(results)

    st.markdown("**Model comparison** (macro F1, average of 25 cross-validation tests)")
    comparison = pd.read_csv("data/model_comparison.csv")
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.image("charts/model_comparison.png")
    st.caption("Reading meaning (embeddings) beats every word-counting model by about +0.07 to +0.09. The three word "
               "models are tied with each other. The TF-IDF model is kept because it can show which words drove a "
               "prediction (the 'Why?' table in the 🤖 tab).")

    st.markdown("**Learning curve:** would more labels help?")
    st.dataframe(pd.read_csv("data/learning_curve.csv"), hide_index=True, width="stretch")
    st.image("charts/learning_curve.png")
    st.caption("The word model slows down after about 70 labels (0.65 at 120). The meaning model starts higher "
               "(0.56 with only 24 labels) and is still rising at 120 (0.74), so more labels would keep helping it.")

    st.markdown("**Error analysis of the word model (TF-IDF):** where does it go wrong? "
                "(each labelled article predicted by a model that never saw it)")
    mix_ups = pd.read_csv("data/error_mixups.csv")
    error_table = pd.read_csv("data/error_analysis.csv")
    st.caption(f"{len(error_table)} mistakes out of 150. Most common mix-ups:")
    st.dataframe(mix_ups.head(5), hide_index=True, width="stretch")
    st.caption("The model's most confident mistakes, and the words that misled it:")
    st.dataframe(error_table.head(10), hide_index=True, width="stretch")
    st.caption("Main lesson: the model counts words and doesn't understand the sentence. 'Paper leak' drowns out the "
               "real action (suspended, bail, cancelled), and 'CJP' also means the Chief Justice of Pakistan.")
