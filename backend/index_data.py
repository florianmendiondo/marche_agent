import os
import re
import chromadb
import unicodedata
from clean_chunk import chunk_text
from extractors import extract_pdf, extract_docx
from mistralai import Mistral, UserMessage
from typing import List  


# --- Chemins ---
CHROMA_PATH = "chroma_db"
#DATA_PATH = r"C:\Users\FMendiondo.adm\OneDrive - UBY\Bureau\python test\proc-memo-mvp\data\raw"
MY_API_KEY = "zCRh11sutA5NY7p9FFySkv7fDbkGZpRt"
MODEL_MISTRAL = "mistral-small"

# --- Fonction de normalisation des tags ---
def normalize_tag(tag: str) -> str:
    """Normalise un tag en majuscules, sans accents ni caractères spéciaux."""
    tag = tag.upper()  # Convertit en majuscules
    tag = ''.join(c for c in unicodedata.normalize('NFD', tag) if unicodedata.category(c) != 'Mn')  # Enlève les accents
    tag = re.sub(r'[^A-Z0-9]+', ' ', tag)  # Remplace les caractères spéciaux par des espaces
    return tag.strip()

# --- Découpage strict en chunks de taille fixe ---
def force_chunk_text(text: str, max_chars=1000, overlap=150):
    """Découpe le texte en chunks de taille fixe avec chevauchement."""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_chars, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if (end - overlap) > start else end
    return chunks

# --- Extraction du marché via LLM ---
def detect_market_llm(text_snippet: str):
    client = Mistral(api_key=MY_API_KEY)
    prompt = f"""
    Tu es un expert spécialisé dans l'extraction de noms de marchés/projets techniques.
    **Règles absolues** :
    1. **Interdiction absolue** de commenter ta réponse.
    2. Extrais **UNIQUEMENT** le nom du marché ou projet principal du texte ci-dessous.
    3. ELIMINE les mots qui te paraissent communs, PRIVILEGIE les lieux, noms propres, etc
    4. Réponds **UNIQUEMENT** avec les mots du nom, en MAJUSCULES, séparés par des espaces.
    5. Si le nom contient des acronymes, chiffres ou noms de lieux, conserve-les.
    Texte :
    {text_snippet}
    """
    messages = [UserMessage(role="user", content=prompt)]
    response = client.chat.complete(model=MODEL_MISTRAL, messages=messages)
    llm_output = response.choices[0].message.content.strip()

# Nettoyage renforcé des tags extraits
    def clean_tag(tag: str) -> str:
        tag = tag.replace("’", " ").replace("'", " ").replace("-", " ").replace("_", " ")
        tag = re.sub(r'[\\/<>]', ' ', tag)
        tag = re.sub(r'[^\w\s]', '', tag)  # Garde lettres, chiffres, espaces
        tag = re.sub(r'\s+', ' ', tag).strip()
        return normalize_tag(tag)  # Normalisation stricte : majuscule + sans accents

    # Extraction et séparation des mots composés
    words = []
    for w in re.split(r'\s+', llm_output):
        if w.strip():
            # Remplacer les séparateurs (tirets, underscores) par des espaces
            w_cleaned = w.replace("'", " ").replace("’", " ").replace("-", " ").replace("_", " ")
            # Séparer les mots composés (ex: "PEMGÈZE" → ["PEM", "GÈZE"])
            parts = w_cleaned.split()
            for p in parts:
                cleaned = clean_tag(p)
                if cleaned and len(cleaned) > 1:  # Ignore les mots trop courts
                    words.append(cleaned)

    # Suppression des doublons et des stopwords
    STOPWORDS_EXTENDED = STOPWORDS_EXTENDED = {
        "LE", "LA", "DE", "DES", "ET", "EST", "LOT", "OPÉRATION", "PROJET", "MARCHÉ", "TRAVAUX", "DANS", "SUR", "POUR",
        "PAR", "AVEC", "NOM", "DU", "AUX", "LES", "UN", "UNE", "IL", "ELLE", "CE", "CET", "CETTE", "CES", "EN", "AU",
        "A", "DES", "LES", "SE", "PAS", "PLUS", "NE", "QUE", "SANS", "SOUS", "OÙ", "OR", "NI", "CAR", "MAIS", "DONT",
        "CELA", "TOUT", "FAIRE", "DIRE", "VOIR", "SAVOIR", "POUVOIR", "VOULOIR", "DEVOIR", "PRENDRE", "DONNER",
        "ALORS", "COMME", "BIEN", "PEU", "PLUS", "MOINS", "TRÈS", "TROP", "ASSEZ", "AUSSI", "SI", "QUAND", "COMMENT",
        "POURQUOI", "QUOI", "LAQUELLE", "LEQUEL", "LESQUELS", "CEUX", "CELLE", "CELLES", "CELUI", "CEUX-LÀ",
        "CELLE-LÀ", "CHACUN", "PLUSIEURS", "DIFFÉRENTS", "NOUVEAU", "ANCIEN", "BON", "MEILLEUR", "PIRE",
        "HAUT", "GRAND", "PETIT", "LONG", "LARGE", "VASTE", "PREMIER", "DERNIER", "SEUL", "MEME", "PROPRE",
        "TOUT", "TOUTE", "TOUS", "TOUTES", "PLUSIEURS", "QUELQUE", "QUELQUES", "AUTRE", "AUTRES", "CERTAIN",
        "CERTAINE", "CERTAINS", "CERTAINES", "DIFFÉRENT", "DIFFÉRENTE", "DIFFÉRENTS", "VARIÉ", "VARIÉE", "VARIÉS",
        "DIVERS", "DIVERSE", "DIVERS", "TEL", "TELLE", "TELS", "TELLES", "MEM", "MEMES", "AUTANT", "AUSSI",
        "ENCORE", "TOUJOURS", "JAMAIS", "DÉJÀ", "BIENTÔT", "TÔT", "TARD", "VITE", "LENT", "RAPIDE",
        "ENSUITE", "PUIS", "DEPUIS", "JUSQUÀ", "AVANT", "APRÈS", "PENDANT", "ENFIN", "FINALEMENT", "SOUDAIN",
        "TOUT À COUP", "ENSEMBLE", "SEUL", "SEULEMENT", "UNIQUEMENT", "SIMPLEMENT", "JUSTE", "VOILÀ", "VOICI",
        "CEST", "IL Y A", "IL EXISTE", "ON TROUVE", "ON A", "ON VOIT", "ON PEUT", "ON DOIT", "IL FAUT", "IL EST",
        "CEST", "IL SAGIT", "NOUS AVONS", "VOUS AVEZ", "ILS ONT", "ELLES ONT", "NOUS SOMMES", "VOUS ÊTES",
        "ILS SONT", "ELLES SONT", "JE SUIS", "TU ES", "IL EST", "ELLE EST", "NOUS SOMMES", "VOUS ÊTES", "ILS SONT",
        "ELLES SONT", "JE VAIS", "TU VAS", "IL VA", "ELLE VA", "NOUS ALLONS", "VOUS ALLEZ", "ILS VONT", "ELLES VONT",
        "NOTE", "HAVE", "EXTRACTED", "ALL", "RELEVANT", "PHRASES", "COMBINED", "THEM", "TO", "FORM", "FINAL",
        "RESPONSE", "PROJECT", "NAME", "APPEARS", "BE", "SIMPLY", "AS", "REPEATED", "SEEMS", "REDUNDANT",
        "LIGNE", "STATION", "SECTION", "CENTRALE", "OUVRAGES", "VAL", "STRUCTURE", "PHASE", "ETAPES", "HMC", "PROCEDURE", "MARCHE", "OPERATION", "OPERATIONS",     "a","ai","ait","ainsi","alors","apres","assez","au","aucun","aucune",
        "aujourdhui","aussi","autre","autres","avant","avec","avoir",
        "bon","car","ce","cela","celle","celles","celui","cent","cependant",
        "ces","cet","cette","chacun","chaque","chez","ci","comme","comment",
        "dans","de","des","du","donc","dos","deja","dire","doit","dont",
        "elle","elles","en","encore","enfin","entre","environ","est","et","etc",
        "eux","faire","fais","fait","fois","font","fut",
        "grand","grace","gros","guere",
        "haut","hors",
        "ici","il","ils","ainsi","j","je","jusqu","juste",
        "la","le","les","leur","leurs","lui",
        "mais","malgre","me","meme","mes","moi","mon","ma",
        "ne","ni","non","nos","notre","nous",
        "on","ont","ou","ou",
        "par","parce","pas","peu","peut","plus","plusieurs","pour","pourquoi","puis",
        "quand","que","quel","quelle","quelles","quels","qui","quoi",
        "sa","sans","se","sera","ses","si","sinon","soi","soit","son","sont","sous","sur",
        "ta","te","tes","toi","ton","tous","tout","toute","toutes",
        "tres","trop","tu",
        "un","une","uns","unes","vos","votre","vous",
        "B", "C", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Z"
    }

    # Filtrer les stopwords et normaliser (accents supprimés, majuscules)
    unique_tags = []
    seen = set()
    for w in words:
        cleaned = clean_tag(w)  # Nettoyage + normalisation
        if cleaned and cleaned not in STOPWORDS_EXTENDED and cleaned not in seen:
            seen.add(cleaned)
            unique_tags.append(cleaned)

    print(f"LLM output brut : {llm_output}")
    print(f"Tags extraits (normalisés et séparés) : {unique_tags}")
    return unique_tags

# --- Indexation ---
def index_documents(file_paths: List[str]):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(name="proc_memo")
        print("Collection existante supprimée !")
    except:
        pass

    collection = client.get_or_create_collection(name="proc_memo")
    print(f"Collection 'proc_memo' prête.\n")

    total_chunks = 0
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️ Fichier introuvable : {file_path}")
            continue

        if file_path.endswith(".pdf"):
            text = extract_pdf(file_path)
        elif file_path.endswith(".docx"):
            text = extract_docx(file_path)
        else:
            print(f"⚠️ Format non supporté : {file_path}")
            continue
    # for filename in os.listdir(file_paths):
    #     if filename.startswith("~$"):
    #         continue

    #     file_path = os.path.join(file_paths, filename)
    #     if filename.endswith(".pdf"):
    #         text = extract_pdf(file_path)
    #     elif filename.endswith(".docx"):
    #         text = extract_docx(file_path)
    #     else:
    #         continue

        # --- Détection du marché ---
        snippet = text[:1000]
        market_tags = detect_market_llm(snippet)

        # --- Découpage en chunks ---
        chunks = force_chunk_text(text, max_chars=1000)
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": os.path.basename(file_path),
                "page": i // 2,
                "chunk_position": i,
                "tags": ";".join(market_tags)  # Tags normalisés, séparés et uniques
            }
            collection.add(
                documents=[chunk],
                metadatas=[metadata],
                ids=[f"{os.path.basename(file_path)}_{i}"]
            )
            print(f"[{os.path.basename(file_path)} - Chunk {i}] Tags : {market_tags} | Taille : {len(chunk)} caractères")

        total_chunks += len(chunks)
        print(f"{os.path.basename(file_path)} : {len(chunks)} chunks ajoutés.\n")

    print(f"✅ Indexation terminée. Total chunks ajoutés : {total_chunks}")
    return collection

if __name__ == "__main__":
    collection = index_documents()

# import os
# import re
# import chromadb
# import unicodedata
# from backend.clean_chunk import chunk_text
# from backend.extractors import extract_pdf, extract_docx
# from mistralai import Mistral, UserMessage

# # --- Chemins ---
# CHROMA_PATH = "chroma_db"
# DATA_PATH = r"C:\Users\FMendiondo.adm\OneDrive - UBY\Bureau\python test\proc-memo-mvp\data\raw"

# MY_API_KEY = "zCRh11sutA5NY7p9FFySkv7fDbkGZpRt"
# MODEL_MISTRAL = "mistral-small"

# def normalize_tag(tag: str) -> str:
#     tag = tag.upper()  # Majuscules pour tous les tags
#     tag = ''.join(c for c in unicodedata.normalize('NFD', tag) if unicodedata.category(c) != 'Mn')
#     tag = re.sub(r'[^A-Z0-9]+', ' ', tag)
#     return tag.strip()

# # --- Découpage strict en chunks de taille fixe ---
# def force_chunk_text(text: str, max_chars=1500, overlap=300):
#     """
#     Découpe le texte en chunks de max_chars caractères avec chevauchement.
#     overlap : nombre de caractères du chunk précédent à réutiliser dans le suivant.
#     """
#     chunks = []
#     start = 0
#     text_length = len(text)

#     while start < text_length:
#         end = min(start + max_chars, text_length)
#         chunk = text[start:end]
#         chunks.append(chunk)
#         # On recule de "overlap" caractères pour garder le contexte
#         start = end - overlap if (end - overlap) > start else end

#     return chunks


# # --- Extraction du marché via LLM ---
# def detect_market_llm(text_snippet: str):
#     client = Mistral(api_key=MY_API_KEY)

#     # Prompt ultra-strict pour éviter les artefacts et commentaires
#     prompt = f"""
#     Tu es un expert spécialisé dans l'extraction de noms de marchés/projets techniques.
#     **Règles absolues** :
#     1. **Interdiction absolue** de commenter ta réponse.  
#     2. Extrais **UNIQUEMENT** le nom du marché ou projet principal du texte ci-dessous.
#     3. Réponds **UNIQUEMENT** avec les mots du nom, en MAJUSCULES, séparés par des espaces.
#     4. Si le nom contient des acronymes, chiffres ou noms de lieux, conserve-les.

#     Texte :
#     {text_snippet}
#     """

#     messages = [UserMessage(role="user", content=prompt)]
#     response = client.chat.complete(model=MODEL_MISTRAL, messages=messages)
#     llm_output = response.choices[0].message.content.strip()

#     # Nettoyage renforcé
#     def clean_tag(tag: str) -> str:
#         # Supprimer les artefacts (caractères spéciaux, tokens de fin, etc.)
#         tag = tag.replace("’", " ").replace("'", " ")  # Remplace apostrophes par espace
#         tag = re.sub(r'[\\/<>]', ' ', tag)
#         tag = re.sub(r'[^\w\s-]', '', tag)  # Garde seulement lettres, chiffres, espaces et tirets
#         tag = re.sub(r'\s+', ' ', tag)  # Remplacer les espaces multiples
#         return tag.strip().upper()

#     # Découper et nettoyer chaque mot, splitter si nécessaire
#     words = []
#     for w in re.split(r'\s+', llm_output):
#         if w.strip():
#             # Remplacer apostrophes ET tirets par des espaces pour scinder les mots composés
#             w_cleaned = w.replace("'", " ").replace("’", " ").replace("-", " ").replace("_", " ")
#             # Split sur espaces internes après remplacement
#             parts = w_cleaned.split()
#             for p in parts:
#                 cleaned = clean_tag(p)
#                 if cleaned and len(cleaned) > 1:
#                     words.append(cleaned)

#     # Filtrer les stopwords et termes génériques
#     STOPWORDS_EXTENDED = {
#         "LE", "LA", "DE", "DES", "ET", "EST", "LOT", "OPÉRATION", "PROJET", "MARCHÉ", "TRAVAUX", "DANS", "SUR", "POUR",
#         "PAR", "AVEC", "NOM", "DU", "AUX", "LES", "UN", "UNE", "IL", "ELLE", "CE", "CET", "CETTE", "CES", "EN", "AU",
#         "A", "DES", "LES", "SE", "PAS", "PLUS", "NE", "QUE", "SANS", "SOUS", "OÙ", "OR", "NI", "CAR", "MAIS", "DONT",
#         "CELA", "TOUT", "FAIRE", "DIRE", "VOIR", "SAVOIR", "POUVOIR", "VOULOIR", "DEVOIR", "PRENDRE", "DONNER",
#         "ALORS", "COMME", "BIEN", "PEU", "PLUS", "MOINS", "TRÈS", "TROP", "ASSEZ", "AUSSI", "SI", "QUAND", "COMMENT",
#         "POURQUOI", "QUOI", "LAQUELLE", "LEQUEL", "LESQUELS", "CEUX", "CELLE", "CELLES", "CELUI", "CEUX-LÀ",
#         "CELLE-LÀ", "CHACUN", "PLUSIEURS", "DIFFÉRENTS", "NOUVEAU", "ANCIEN", "BON", "MEILLEUR", "PIRE",
#         "HAUT", "GRAND", "PETIT", "LONG", "LARGE", "VASTE", "PREMIER", "DERNIER", "SEUL", "MEME", "PROPRE",
#         "TOUT", "TOUTE", "TOUS", "TOUTES", "PLUSIEURS", "QUELQUE", "QUELQUES", "AUTRE", "AUTRES", "CERTAIN",
#         "CERTAINE", "CERTAINS", "CERTAINES", "DIFFÉRENT", "DIFFÉRENTE", "DIFFÉRENTS", "VARIÉ", "VARIÉE", "VARIÉS",
#         "DIVERS", "DIVERSE", "DIVERS", "TEL", "TELLE", "TELS", "TELLES", "MEM", "MEMES", "AUTANT", "AUSSI",
#         "ENCORE", "TOUJOURS", "JAMAIS", "DÉJÀ", "BIENTÔT", "TÔT", "TARD", "VITE", "LENT", "RAPIDE",
#         "ENSUITE", "PUIS", "DEPUIS", "JUSQUÀ", "AVANT", "APRÈS", "PENDANT", "ENFIN", "FINALEMENT", "SOUDAIN",
#         "TOUT À COUP", "ENSEMBLE", "SEUL", "SEULEMENT", "UNIQUEMENT", "SIMPLEMENT", "JUSTE", "VOILÀ", "VOICI",
#         "CEST", "IL Y A", "IL EXISTE", "ON TROUVE", "ON A", "ON VOIT", "ON PEUT", "ON DOIT", "IL FAUT", "IL EST",
#         "CEST", "IL SAGIT", "NOUS AVONS", "VOUS AVEZ", "ILS ONT", "ELLES ONT", "NOUS SOMMES", "VOUS ÊTES",
#         "ILS SONT", "ELLES SONT", "JE SUIS", "TU ES", "IL EST", "ELLE EST", "NOUS SOMMES", "VOUS ÊTES", "ILS SONT",
#         "ELLES SONT", "JE VAIS", "TU VAS", "IL VA", "ELLE VA", "NOUS ALLONS", "VOUS ALLEZ", "ILS VONT", "ELLES VONT",
#         "NOTE", "HAVE", "EXTRACTED", "ALL", "RELEVANT", "PHRASES", "COMBINED", "THEM", "TO", "FORM", "FINAL",
#         "RESPONSE", "PROJECT", "NAME", "APPEARS", "BE", "SIMPLY", "AS", "REPEATED", "SEEMS", "REDUNDANT",
#         "LIGNE", "STATION", "SECTION", "CENTRALE", "OUVRAGES", "VAL", "STRUCTURE", "PHASE", "ETAPES", "HMC", "PROCEDURE", "MARCHE", "OPERATION", "OPERATIONS",     "a","ai","ait","ainsi","alors","apres","assez","au","aucun","aucune",
#         "aujourdhui","aussi","autre","autres","avant","avec","avoir",
#         "bon","car","ce","cela","celle","celles","celui","cent","cependant",
#         "ces","cet","cette","chacun","chaque","chez","ci","comme","comment",
#         "dans","de","des","du","donc","dos","deja","dire","doit","dont",
#         "elle","elles","en","encore","enfin","entre","environ","est","et","etc",
#         "eux","faire","fais","fait","fois","font","fut",
#         "grand","grace","gros","guere",
#         "haut","hors",
#         "ici","il","ils","ainsi","j","je","jusqu","juste",
#         "la","le","les","leur","leurs","lui",
#         "mais","malgre","me","meme","mes","moi","mon","ma",
#         "ne","ni","non","nos","notre","nous",
#         "on","ont","ou","ou",
#         "par","parce","pas","peu","peut","plus","plusieurs","pour","pourquoi","puis",
#         "quand","que","quel","quelle","quelles","quels","qui","quoi",
#         "sa","sans","se","sera","ses","si","sinon","soi","soit","son","sont","sous","sur",
#         "ta","te","tes","toi","ton","tous","tout","toute","toutes",
#         "tres","trop","tu",
#         "un","une","uns","unes","vos","votre","vous",
#         "B", "C", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Z"
#     }

#     # Filtrer les mots
#     tags = [
#         w for w in words
#         if w not in STOPWORDS_EXTENDED and (
#             w.isupper() or  # Acronyme ou mot en majuscules
#             any(c.isdigit() for c in w) or  # Contient un chiffre
#             len(w) > 3  # Longueur minimale
#         )
#     ]

#     # Supprimer les doublons
#     seen = set()
#     unique_tags = []
#     for w in tags:
#         if w not in seen:
#             seen.add(w)
#             unique_tags.append(w)

#     print(f"LLM output brut : {llm_output}")
#     print(f"Tags extraits (après filtre) : {unique_tags}")
#     return unique_tags

# # --- Indexation ---
# def index_documents():
#     client = chromadb.PersistentClient(path=CHROMA_PATH)

#     try:
#         client.delete_collection(name="proc_memo")
#         print("Collection existante supprimée !")
#     except:
#         pass

#     collection = client.get_or_create_collection(name="proc_memo")
#     print(f"Collection 'proc_memo' prête.\n")

#     total_chunks = 0

#     for filename in os.listdir(DATA_PATH):
#         if filename.startswith("~$"):
#             continue
#         file_path = os.path.join(DATA_PATH, filename)

#         if filename.endswith(".pdf"):
#             text = extract_pdf(file_path)
#         elif filename.endswith(".docx"):
#             text = extract_docx(file_path)
#         else:
#             continue

#         # --- Détection du marché ---
#         snippet = text[:1000]
#         market_tags = detect_market_llm(snippet)

#         # --- Découpage en chunks ~1500 caractères ---
#         chunks = force_chunk_text(text, max_chars=1500)

#         for i, chunk in enumerate(chunks):
#             metadata = {
#                 "source": filename,
#                 "page": i // 2,  # Approximation de page
#                 "chunk_position": i,  # Position dans le document
#                 "tags": ";".join(market_tags)  # Convertit la liste en chaîne   
#                          }
#             collection.add(
#                 documents=[chunk],
#                 metadatas=[metadata],
#                 ids=[f"{filename}_{i}"]
#             )
#             print(f"[{filename} - Chunk {i}] Tags : {market_tags} | Taille : {len(chunk)} caractères")

#         total_chunks += len(chunks)
#         print(f"{filename} : {len(chunks)} chunks ajoutés.\n")

#     print(f"✅ Indexation terminée. Total chunks ajoutés : {total_chunks}")
#     return collection


# if __name__ == "__main__":
#     collection = index_documents()
