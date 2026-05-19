import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalise un message sans supprimer son sens."""
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+", " lien ", text)
    text = re.sub(r"@\w+", " mention ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text
