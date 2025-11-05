import pandas as pd
from collections import Counter
from typing import Tuple, List

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def infer_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [c.lower() for c in df.columns]
    label = next((c for c in cols if "label" in c or "target" in c), cols[0])
    text = next((c for c in cols if "text" in c or "message" in c), cols[-1])
    return label, text

def token_topn(series: pd.Series, topn: int = 20) -> List:
    counter = Counter(" ".join(series.astype(str)).split())
    return counter.most_common(topn)
