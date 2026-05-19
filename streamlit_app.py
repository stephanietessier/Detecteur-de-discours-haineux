import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import load_dataset
from src.moderation_model import predict_message, train_model

st.set_page_config(page_title="Détecteur de discours haineux", page_icon="🛡️", layout="wide")
st.title("🛡️ Détecteur de discours haineux — prototype responsable")
st.write("Prototype éducatif : l'IA signale les risques, mais la décision finale doit rester humaine.")

@st.cache_resource
def get_model():
    df = load_dataset()
    model, report, matrix, classes = train_model(df)
    return df, model, report, matrix, classes


df, model, report, matrix, classes = get_model()

with st.sidebar:
    st.header("Données")
    st.write(f"Messages fictifs : {len(df)}")
    st.write("Catégories :")
    for cls in classes:
        st.write(f"- {cls}")

message = st.text_area("Message à analyser", "Je ne suis pas d'accord, mais je veux comprendre ton point de vue.")

if st.button("Analyser"):
    result = predict_message(model, message)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Résultat")
        st.metric("Catégorie prédite", result.label)
        st.metric("Confiance", f"{result.confidence:.0%}")
        if result.needs_human_review:
            st.warning("Confiance faible : révision humaine recommandée.")
        else:
            st.success("Confiance suffisante pour un tri initial, avec supervision humaine.")
    with col2:
        st.subheader("Probabilités")
        prob_df = pd.DataFrame(
            {"catégorie": list(result.probabilities.keys()), "probabilité": list(result.probabilities.values())}
        ).sort_values("probabilité", ascending=False)
        st.bar_chart(prob_df.set_index("catégorie"))

    st.subheader("Termes influents")
    if result.explanation_terms:
        st.write(", ".join(result.explanation_terms))
    else:
        st.write("Aucun terme explicatif fort détecté.")

with st.expander("Voir les données fictives"):
    st.dataframe(df)

with st.expander("Rapport d'évaluation"):
    st.text(report)
