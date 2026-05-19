from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "messages_moderation_fictifs.csv"
RANDOM_STATE = 42
LABELS = ["acceptable", "insulte", "harcelement", "haine", "menace"]
REVIEW_THRESHOLD = 0.62
