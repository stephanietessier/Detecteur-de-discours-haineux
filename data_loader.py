import pandas as pd
from .config import DATA_PATH


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Charge le jeu de données fictif et vérifie les colonnes attendues."""
    df = pd.read_csv(path)
    expected = {"message", "label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")
    df = df.dropna(subset=["message", "label"]).copy()
    df["message"] = df["message"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    return df
