# backend/generator.py
import chromadb
from mistralai import Mistral, UserMessage
import unicodedata
import re
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Clé API et modèle ---
MY_API_KEY = os.getenv("MISTRAL_API_KEY")  # Clé API Mistral (à sécuriser en production)
MODEL_MISTRAL = "mistral-large-2411"

# Liste des mots trop génériques dans les marchés publics
STOPWORDS_TAGS = {
    "marche","marches","appel","appels","offre","offres","appel d’offre","appels d’offres",
    "dossier","dossiers","consultation","consultations",
    "contrat","contrats","convention","conventions","accord","accords",
    "procedure","procedures","candidature","candidatures",
    "lot","lots","tranche","tranches","phase","phases",
    "prestation","prestations","service","services",
    "travail","travaux","chantier","chantiers","projet","projets",
    "operation","operations","opération","opérations","realisation","realisations",
    "construction","constructions","rehabilitation","rehabilitations",
    "renovation","renovations","reparation","reparations",
    "entretien","maintenance","maintenances",
    "exploitation","exploitations","fourniture","fournitures",
    "livraison","livraisons",
    "amenagement","amenagements","equipement","equipements",
    "installation","installations","mise en oeuvre","mise en service",
    "demolition","demolitions","deconstruction","deconstructions",
    "extension","extensions","agrandissement","agrandissements",
    "etude","etudes","conception","conceptions","ingenierie","maitrise","maîtrise",
    "maitrise d’oeuvre","maitrise d’ouvrage","assistance",
    "audit","audits","expertise","expertises",
    "developpement","developments","elaboration","elaborations",
    "preparation","preparations","coordination","coordinations",
    "gestion","gestions","suivi","suivis",
    "planification","planifications","organisation","organisations",
    "execution","executions","production","productions",
    "activite","activites","mission","missions","objet","objets",
    "batiment","batiments","ouvrage","ouvrages","infrastructure","infrastructures",
    "reseau","reseaux","systeme","systemes",
    "logiciel","logiciels","plateforme","plateformes",
    "materiel","materiaux","fournitures","B", "C", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Z"
}

def normalize_tag(tag: str) -> str:
    """Supprime accents, met en majuscule, enlève caractères spéciaux"""
    tag = tag.upper()
    tag = ''.join(c for c in unicodedata.normalize('NFD', tag) if unicodedata.category(c) != 'Mn')
    tag = re.sub(r'[^A-Z0-9]+', ' ', tag)
    return tag.strip()

def extract_tags_from_prompt_llm(query: str):
    client = Mistral(api_key=MY_API_KEY)
    prompt = f"""
    Tu es un extracteur de tags.
    Analyse la requête utilisateur et retourne UNIQUEMENT les mots discriminants
    (les termes rares, lieux, numéros de ligne, acronymes).
    Retourne la réponse sous forme de liste JSON en MAJUSCULES.
    Défense ABSOLUE de mettre "JSON" dans ta liste.

    Question : {query}
    """
    messages = [UserMessage(role="user", content=prompt)]
    response = client.chat.complete(model=MODEL_MISTRAL, messages=messages)

    tags_text = response.choices[0].message.content.strip()

    # --- 1. Essaye de parser directement en JSON ---
    try:
        raw_tags = json.loads(tags_text)
        if not isinstance(raw_tags, list):
            raise ValueError("Format inattendu")
    except Exception:
        # --- 2. Fallback : coupe à la main si pas du JSON valide ---
        raw_tags = re.findall(r"[A-ZÀ-ÖØ-Ý][A-ZÀ-ÖØ-Ý0-9\-]+", tags_text.upper())

    # --- 3. Normalisation stricte et filtrage ---
    tags = []
    seen = set()
    for t in raw_tags:
        norm = normalize_tag(t)
        if norm and norm not in STOPWORDS_TAGS and len(norm) > 2 and norm not in seen:
            seen.add(norm)
            tags.append(norm)

    return tags

# --- Charger la collection ChromaDB ---
def load_chroma_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection("proc_memo")

# --- Recherche et filtrage des chunks avec recherche partielle ---
def get_relevant_chunks(query: str, top_k: int = 15):
    collection = load_chroma_collection()
    print("\n=== ANALYSE DES CHUNKS ===")
    print(f"Requête: {query}")

    # --- Extraction des tags ---
    tags = extract_tags_from_prompt_llm(query)
    print(f"Tags extraits: {tags}")

    # --- Récupérer tous les chunks (documents + métadatas + ids) ---
    all_results = collection.get(include=["metadatas", "documents", "embeddings"])
    all_ids = all_results["ids"]
    all_metas = all_results["metadatas"]
    all_docs = all_results["documents"]
    all_embs = all_results["embeddings"]

    # --- Étape 1 : pré-sélection par tags ---
    preselected = []
    for id_, meta, doc, emb in zip(all_ids, all_metas, all_docs, all_embs):
        chunk_tags = meta.get("tags", "").split(";")
        if any(t in chunk_tags for t in tags):
            preselected.append({
                "id": id_,
                "document": doc,
                "metadata": meta,
                "embedding": emb
            })

    # --- Étape 2a : tags matchés ---
    if preselected:
        print(f"✅ {len(preselected)} chunks correspondent aux tags extraits.")

        # Calcul embedding de la requête
        query_emb = collection._embedding_function([query])[0]

        # Calcul des similarités cosinus
        query_emb = np.array(query_emb).reshape(1, -1)
        preselected_embs = np.array([c["embedding"] for c in preselected])
        similarities = cosine_similarity(query_emb, preselected_embs)[0]

        # Attacher similarités aux chunks
        for chunk, sim in zip(preselected, similarities):
            chunk["similarity_score"] = float(sim)

        # Tri par similarité
        preselected.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_chunks = preselected[:top_k]

    # --- Étape 2b : aucun tag ne correspond ---
    else:
        print("⚠️ Aucun tag ne concorde. Réponse à partir des documents envoyés.")

        # Recherche vectorielle pure
        query_results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        top_chunks = []
        for doc, meta, dist in zip(
            query_results["documents"][0],
            query_results["metadatas"][0],
            query_results["distances"][0]
        ):
            top_chunks.append({
                "document": doc,
                "metadata": meta,
                "similarity_score": 1 / (1 + dist)
            })

    # --- Résumé final ---
    print(f"\nTop chunks sélectionnés: {len(top_chunks)}")
    for i, chunk in enumerate(top_chunks):
        meta = chunk["metadata"]
        print(f"TOP {i+1} | Score: {chunk.get('similarity_score', 'N/A'):.4f} | "
              f"Source: {meta.get('source', 'inconnu')} | Page: {meta.get('page', 'inconnu')}")

    return [chunk["document"] for chunk in top_chunks]
# def get_relevant_chunks(query: str, top_k: int = 5):
#     collection = load_chroma_collection()
#     print("\n=== ANALYSE DES CHUNKS ===")
#     print(f"Requête: {query}")

#     # --- Extraction des tags ---
#     tags = extract_tags_from_prompt_llm(query)
#     print(f"Tags extraits: {tags}")

#     # --- Cas 1 : Aucun tag → Recherche vectorielle pure ---
#     if not tags:
#         print("⚠️ Aucun tag extrait. Recherche vectorielle pure.")
#         query_results = collection.query(
#             query_texts=[query],
#             n_results=top_k,
#             include=["documents", "metadatas", "distances"]
#         )
#         scored_chunks = []
#         if "metadatas" in query_results and isinstance(query_results["metadatas"], list):
#             for doc, meta, dist in zip(
#                 query_results["documents"][0],
#                 query_results["metadatas"][0],
#                 query_results["distances"][0]
#             ):
#                 if isinstance(meta, dict):
#                     meta_id = meta.get("id", "unknown_id")
#                 else:
#                     print(f"⚠️ Métadonnées inattendues (type: {type(meta)}): {meta}")
#                     meta_id = "unknown_id"

#                 scored_chunks.append({
#                     "id": meta_id,
#                     "document": doc,
#                     "metadata": meta,
#                     "tag_match_score": 0.0,
#                     "similarity_score": 1 / (1 + dist)
#                 })
#         else:
#             print("⚠️ Structure inattendue pour query_results['metadatas']")
#             return []

#     # --- Cas 2 : Tags extraits → Retourner les chunks pré-sélectionnés (tag_match=1.0) ---
#     else:
#         # # --- Récupérer TOUS les chunks avec leurs métadonnées ---
#         # all_results = collection.get(include=["metadatas", "documents"])
#         # print(f"Structure de all_results: {all_results.keys()}")

#         # if "metadatas" not in all_results:
#         #     print("⚠️ Clé 'metadatas' manquante dans all_results")
#         #     return []

#         # all_metas = all_results["metadatas"]
#         # all_docs = all_results["documents"]

#         # if not isinstance(all_metas, list):
#         #     print(f"⚠️ all_metas n'est pas une liste (type: {type(all_metas)})")
#         #     return []

#         # # --- Étape 1 : Pré-sélection par tags (tag_match = 1.0) ---
#         # preselected_chunks = []
#         # for meta, doc in zip(all_metas, all_docs):
#         #     if not isinstance(meta, dict):
#         #         print(f"⚠️ Métadonnées inattendues (type: {type(meta)}): {meta}")
#         #         continue

#         #     chunk_tags = meta.get("tags", "").split(";")
#         #     if any(t in chunk_tags for t in tags):
#         #         preselected_chunks.append({
#         #             "id": meta.get("id", "unknown_id"),
#         #             "document": doc,
#         #             "metadata": meta,
#         #             "tag_match_score": 1.0
#         #         })

#         # print(f"Chunks pré-sélectionnés (tag_match=1.0): {len(preselected_chunks)}")

#         # # --- Affichage détaillé des chunks pré-sélectionnés ---
#         # print("\n=== DÉTAILS DES CHUNKS PRÉ-SÉLECTIONNÉS ===")
#         # for i, chunk in enumerate(preselected_chunks):
#         #     meta = chunk["metadata"]
#         #     source = meta.get("source", "inconnu")
#         #     page = meta.get("page", "inconnu")
#         #     tags = meta.get("tags", "aucun")
#         #     doc_id = chunk["id"]
#         #     print(f"Chunk {i+1} | ID: {doc_id} | Source: {source} | Page: {page} | Tags: {tags}")
#         #     print(f"Extrait: {chunk['document'][:100]}...")
#         #     print("---")

#         # # --- Retourner directement les chunks pré-sélectionnés (sans calcul de similarité) ---
#         # top_chunks = preselected_chunks[:top_k]

#         # print(f"\nTop chunks sélectionnés (tag_match=1.0): {len(top_chunks)}")
#         # for i, chunk in enumerate(top_chunks):
#         #     meta = chunk["metadata"]
#         #     print(f"TOP {i+1} | Source: {meta.get('source', 'inconnu')} | Page: {meta.get('page', 'inconnu')}")

#         # return [chunk["document"] for chunk in top_chunks]

#         # --- Récupérer les IDs des chunks pré-sélectionnés ---
#         preselected_ids = [c["id"] for c in preselected_chunks]

#         # --- Si aucun chunk pré-sélectionné, retourner une liste vide ---
#         if not preselected_ids:
#             print("⚠️ Aucun chunk pré-sélectionné.")
#             return []

#         # --- Effectuer une requête vectorielle sur TOUS les chunks, mais filtrer les résultats ---
#         query_results = collection.query(
#             query_texts=[query],
#             n_results=len(preselected_ids),  # On demande assez de résultats
#             include=["documents", "metadatas", "distances"]
#         )

#         # --- Filtrer les résultats pour ne garder que les chunks pré-sélectionnés ---
#         scored_chunks = []
#         for doc, meta, dist in zip(
#             query_results["documents"][0],
#             query_results["metadatas"][0],
#             query_results["distances"][0]
#         ):
#             if not isinstance(meta, dict):
#                 continue

#             doc_id = meta.get("id", "unknown_id")
#             if doc_id in preselected_ids:  # On ne garde QUE les chunks pré-sélectionnés
#                 scored_chunks.append({
#                     "id": doc_id,
#                     "document": doc,
#                     "metadata": meta,
#                     "tag_match_score": 1.0,
#                     "similarity_score": 1 / (1 + dist)  # Convertir la distance en score
#                 })

#         # --- Tri des chunks pré-sélectionnés par similarité ---
#         if scored_chunks:
#             scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
#             top_chunks = scored_chunks[:top_k]

#             print(f"\nTop chunks sélectionnés (tag_match=1.0, triés par similarité): {len(top_chunks)}")
#             for i, chunk in enumerate(top_chunks):
#                 meta = chunk["metadata"]
#                 print(f"TOP {i+1} | Sim: {chunk['similarity_score']:.4f} | Source: {meta.get('source', 'inconnu')} | Page: {meta.get('page', 'inconnu')}")

#             return [chunk["document"] for chunk in top_chunks]
#         else:
#             print("⚠️ Aucun chunk sélectionné après filtrage.")
#             return []
    
# # def get_relevant_chunks(query: str, top_k: int = 5):
# #     collection = load_chroma_collection()
# #     print("\n=== ANALYSE DES CHUNKS ===")
# #     print(f"Requête: {query}")

# #     # --- Extraction des tags ---
# #     tags = extract_tags_from_prompt_llm(query)
# #     print(f"Tags extraits: {tags}")

# #     # --- Cas 1 : Aucun tag → Recherche vectorielle pure ---
# #     if not tags:
# #         print("⚠️ Aucun tag extrait. Recherche vectorielle pure.")
# #         query_results = collection.query(
# #             query_texts=[query],
# #             n_results=top_k,
# #             include=["documents", "metadatas", "distances"]
# #         )
# #         scored_chunks = []
# #         for doc, meta, dist, id_ in zip(
# #             query_results["documents"][0],
# #             query_results["metadatas"][0],
# #             query_results["distances"][0],
# #             query_results["ids"][0]
# #         ):
# #             scored_chunks.append({
# #                 "id": id_,
# #                 "document": doc,
# #                 "metadata": meta,
# #                 "tag_match_score": 0.0,
# #                 "similarity_score": 1 / (1 + dist)
# #             })

# #     # --- Cas 2 : Tags extraits → Filtrage par tags + recherche vectorielle ---
# #     else:
# #         # --- Récupérer TOUS les chunks avec leurs métadonnées ---
# #         all_results = collection.get(include=["metadatas", "documents"])
# #         all_ids = all_results["ids"]
# #         all_metas = all_results["metadatas"]
# #         all_docs = all_results["documents"]

# #         # --- Pré-sélection par tags ---
# #         preselected_chunks = []
# #         for id_, meta, doc in zip(all_ids, all_metas, all_docs):
# #             chunk_tags = meta.get("tags", "").split(";")
# #             if any(t in chunk_tags for t in tags):
# #                 preselected_chunks.append({
# #                     "id": id_,
# #                     "document": doc,
# #                     "metadata": meta,
# #                     "tag_match_score": 1.0
# #                 })

# #         print(f"Chunks pré-sélectionnés: {len(preselected_chunks)}")

# #         # --- Si aucun chunk ne match → Fallback vectoriel ---
# #         if not preselected_chunks:
# #             print("⚠️ Aucun chunk ne match les tags. Recherche vectorielle pure.")
# #             query_results = collection.query(
# #                 query_texts=[query],
# #                 n_results=top_k,
# #                 include=["documents", "metadatas", "distances"]
# #             )
# #             scored_chunks = []
# #             for doc, meta, dist, id_ in zip(
# #                 query_results["documents"][0],
# #                 query_results["metadatas"][0],
# #                 query_results["distances"][0],
# #                 query_results["ids"][0]
# #             ):
# #                 scored_chunks.append({
# #                     "id": id_,
# #                     "document": doc,
# #                     "metadata": meta,
# #                     "tag_match_score": 0.0,
# #                     "similarity_score": 1 / (1 + dist)
# #                 })

# #         # --- Sinon : Recherche vectorielle sur les chunks pré-sélectionnés ---
# #         else:
# #             preselected_ids = [c["id"] for c in preselected_chunks]
# #             query_results = collection.query(
# #                 query_texts=[query],
# #                 n_results=len(preselected_ids),
# #                 include=["documents", "metadatas", "distances"]
# #             )
# #             scored_chunks = []
# #             for doc, meta, dist, id_ in zip(
# #                 query_results["documents"][0],
# #                 query_results["metadatas"][0],
# #                 query_results["distances"][0],
# #                 query_results["ids"][0]
# #             ):
# #                 if id_ in preselected_ids:
# #                     scored_chunks.append({
# #                         "id": id_,
# #                         "document": doc,
# #                         "metadata": meta,
# #                         "tag_match_score": 1.0,
# #                         "similarity_score": 1 / (1 + dist)
# #                     })

# #     # --- Tri et retour ---
# #     scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
# #     top_chunks = scored_chunks[:top_k]

# #     print(f"\nTop chunks sélectionnés: {len(top_chunks)}")
# #     if not top_chunks:
# #         print("⚠️ Aucun chunk sélectionné.")
# #         return []
# #     else:
# #         return [chunk["document"] for chunk in top_chunks]

# --- Print les chunks pour comprendre ce qui est envoyé au llm ---
def print_chunks_debug(chunks):
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ({len(chunk)} caractères) ---\n")
        print(chunk[:500])  # affiche jusqu'à 500 caractères si chunk trop long

# --- Générer un paragraphe structuré à partir des chunks ---
def generate_paragraph(query, relevant_chunks):
    client_m = Mistral(api_key=MY_API_KEY)
    context = "\n".join(relevant_chunks)
    prompt = f"""
    Contexte : Tu es un assistant expert technique dans le domaine ciblé de la requête utilisateur.
    Consignes :
    1. Base-toi UNIQUEMENT sur les documents fournis ci-dessous.
    2. Structure ta réponse en 2 paragraphes : [Contexte du marché] / [Réponse technique factuelle + extraits].
    3. Cite les sources (ex: "Selon le document intitulé "TITRE", page Y, l'extrait suivant ("...") indique que...").
    Documents :
    {context}
    Question: {query}
    """
    messages = [UserMessage(role="user", content=prompt)]
    response = client_m.chat.complete(model=MODEL_MISTRAL, messages=messages)
    return response.choices[0].message.content

# --- Fonction principale d'usage ---
def generate_from_query(query):
    chunks = get_relevant_chunks(query)
    print_chunks_debug(chunks)
    if not chunks:
        return "Aucun document pertinent trouvé pour cette requête."
    return generate_paragraph(query, chunks)

# --- Test rapide ---
if __name__ == "__main__":
    query = "Décris moi le système d'instrumentation ou les capteurs à mettre en place pour le marché du PEM Geze ? Quelle quantité prévoir ?"
    result = generate_from_query(query)
    print("\n--- Paragraphe généré ---")
    print(result)
