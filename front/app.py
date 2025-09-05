import streamlit as st
import requests
import os

# --- URL du backend ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Configuration de la page ---
st.set_page_config(page_title="Agent d'Analyse de Marchés", layout="wide")
st.title("🔍 Agent d'Analyse de Cahiers des Charges")
st.markdown("""
Cette interface permet d'uploader des documents (PDF/Word) et d'obtenir une analyse structurée des exigences techniques.
""")

# --- État global ---
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "indexing_status" not in st.session_state:
    st.session_state.indexing_status = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# --- Section 1 : Upload ---
st.header("1. Upload des Documents")
uploaded_files = st.file_uploader(
    "Glissez-déposez vos fichiers (PDF/Word) ici",
    accept_multiple_files=True,
    type=["pdf", "docx"]
)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    if st.button("📤 Indexer les documents"):
        with st.spinner("Indexation en cours..."):
            files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
            try:
                response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=240)
                response.raise_for_status()
                st.session_state.indexing_status = f"✅ {len(uploaded_files)} documents indexés avec succès."
                st.success(st.session_state.indexing_status)
                st.json(response.json())
            except requests.exceptions.RequestException as e:
                st.session_state.indexing_status = f"❌ Erreur de connexion : {e}"
                st.error(st.session_state.indexing_status)

# --- Section 2 : Analyse ---
st.header("2. Analyse de la Requête")
query = st.text_area("Posez votre question", height=100)

if st.button("🔍 Analyser la requête"):
    if not query:
        st.warning("Veuillez entrer une question.")
    elif not st.session_state.uploaded_files:
        st.warning("Veuillez d'abord uploader des documents.")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(f"{BACKEND_URL}/analyze", json={"query": query}, timeout=240)
                response.raise_for_status()
                st.session_state.analysis_result = response.json().get("summary", "")
                st.success("Analyse terminée !")
            except requests.exceptions.RequestException as e:
                st.session_state.analysis_result = f"❌ Erreur de connexion : {e}"
                st.error(st.session_state.analysis_result)

# --- Section 3 : Résultats ---
st.header("3. Résultats")
if st.session_state.indexing_status:
    st.subheader("📋 Statut de l'indexation")
    st.code(st.session_state.indexing_status)

if st.session_state.analysis_result:
    st.subheader("📄 Résultat de l'analyse")
    if st.session_state.analysis_result.startswith("❌"):
        st.error(st.session_state.analysis_result)
    else:
        st.markdown(st.session_state.analysis_result)
        st.download_button(
            label="💾 Télécharger en Markdown",
            data=st.session_state.analysis_result,
            file_name="analyse_marche.md",
            mime="text/markdown"
        )

# --- Section 4 : Logs de debug ---
with st.expander("🐞 Logs de debug"):
    st.json({
        "Fichiers uploadés": [f.name for f in st.session_state.uploaded_files] if st.session_state.uploaded_files else None,
        "Statut indexation": st.session_state.indexing_status,
        "Résultat analyse (début)": str(st.session_state.analysis_result)[:200] + "..." if st.session_state.analysis_result else None
    })