import streamlit as st
import requests
import time

# --- Configuration de la page ---
st.set_page_config(page_title="Agent d'Analyse de Marchés", layout="wide")
st.title("🔍 Agent d'Analyse de Cahiers des Charges")
st.markdown("""
Cette interface permet d'uploader des documents (PDF/Word) et d'obtenir une analyse structurée des exigences techniques.
""")

# --- État global pour suivre les étapes ---
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "indexing_status" not in st.session_state:
    st.session_state.indexing_status = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# --- Section 1 : Upload des documents ---
st.header("1. Upload des Documents")
uploaded_files = st.file_uploader(
    "Glissez-déposez vos fichiers (PDF/Word) ici",
    accept_multiple_files=True,
    type=["pdf", "docx"],
    key="file_uploader"
)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    if st.button("📤 Indexer les documents"):
        with st.spinner("Indexation en cours..."):
            st.session_state.indexing_status = "En cours..."
            files = []
            for uploaded_file in uploaded_files:
                files.append(("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)))

            try:
                response = requests.post(
                    "http://localhost:8000/upload",
                    files=files,
                    timeout=30  # Timeout pour éviter les blocages
                )
                if response.status_code == 200:
                    st.session_state.indexing_status = f"✅ Succès : {len(uploaded_files)} documents indexés."
                    st.success(st.session_state.indexing_status)
                    st.json(response.json())
                else:
                    st.session_state.indexing_status = f"❌ Erreur : {response.json().get('detail', 'Inconnu')}"
                    st.error(st.session_state.indexing_status)
            except requests.exceptions.RequestException as e:
                st.session_state.indexing_status = f"❌ Erreur de connexion : {str(e)}"
                st.error(st.session_state.indexing_status)

# --- Section 2 : Analyse de la requête ---
st.header("2. Analyse de la Requête")
query = st.text_area(
    "Posez votre question (ex: 'Quelles sont les méthodes d'auscultation géotechnique exigées ?')",
    height=100,
    key="query_input"
)

if st.button("🔍 Analyser la requête"):
    if not query:
        st.warning("Veuillez entrer une question.")
    else:
        if not st.session_state.uploaded_files:
            st.warning("Veuillez d'abord uploader des documents.")
        else:
            with st.spinner("Analyse en cours..."):
                st.session_state.analysis_result = "En cours..."
                try:
                    response = requests.post(
                        "http://localhost:8000/analyze",
                        json={"query": query},
                        timeout=30
                    )
                    if response.status_code == 200:
                        st.session_state.analysis_result = response.json()["summary"]
                        st.success("Analyse terminée !")
                    else:
                        st.session_state.analysis_result = f"❌ Erreur : {response.json().get('detail', 'Inconnu')}"
                        st.error(st.session_state.analysis_result)
                except requests.exceptions.RequestException as e:
                    st.session_state.analysis_result = f"❌ Erreur de connexion : {str(e)}"
                    st.error(st.session_state.analysis_result)

# --- Section 3 : Affichage des résultats ---
st.header("3. Résultats")
if st.session_state.indexing_status:
    st.subheader("📋 Statut de l'indexation")
    st.code(st.session_state.indexing_status)

if st.session_state.analysis_result and st.session_state.analysis_result != "En cours...":
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

# --- Section 4 : Logs de debug (optionnel) ---
with st.expander("🐞 Voir les logs de debug"):
    st.write("**État actuel :**")
    st.json({
        "Fichiers uploadés": [f.name for f in st.session_state.uploaded_files] if st.session_state.uploaded_files else None,
        "Statut indexation": st.session_state.indexing_status,
        "Résultat analyse": str(st.session_state.analysis_result)[:200] + "..." if st.session_state.analysis_result else None
    })
