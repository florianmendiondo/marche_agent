import re  # importe le module 're' pour les expressions régulières (recherche, remplacement, découpage de texte)
from unidecode import unidecode  # importe la fonction 'unidecode' pour convertir les caractères accentués en caractères ASCII simples

# Fonction pour nettoyer le texte
def clean_text(text: str) -> str:
    """Nettoie le texte brut (espaces multiples, accents normalisés)."""
    t = unidecode(text)  # transforme les caractères accentués en leur équivalent ASCII (é -> e, ü -> u, etc.)
    t = re.sub(r"\s+", " ", t)  # remplace tous les groupes d'espaces, tabulations ou retours à la ligne par un seul espace
    return t.strip()  # supprime les espaces en début et fin de texte

# Fonction pour découper le texte en "chunks"
def chunk_text(text: str, max_chars=500, overlap=150):
    """Découpe le texte en morceaux (chunks) utilisables pour RAG (retrieval-augmented generation)."""
    # Sépare le texte en paragraphes en se basant sur les doubles sauts de ligne et supprime les paragraphes vides
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []  # liste qui contiendra les chunks finaux
    buf = ""  # buffer temporaire pour construire les chunks
    for p in paras:  # parcourt chaque paragraphe
        if len(buf) + len(p) + 1 <= max_chars:  # si le paragraphe peut tenir dans le chunk actuel
            buf = f"{buf}\n{p}" if buf else p  # ajoute le paragraphe au buffer avec un saut de ligne si le buffer n'est pas vide
        else:
            if buf:  # si le buffer contient déjà du texte
                chunks.append(buf)  # ajoute le chunk terminé à la liste
            # prépare le nouveau buffer avec chevauchement (overlap)
            buf = (buf[-overlap:] + "\n" + p).strip()  # conserve les derniers caractères du chunk précédent et ajoute le paragraphe courant
    if buf:  # après la boucle, si le buffer contient du texte
        chunks.append(buf)  # ajoute le dernier chunk
    return chunks  # retourne la liste des chunks
