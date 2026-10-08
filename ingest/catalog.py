import pandas as pd
from core import config


def load_catalog() -> list[dict]:
    df = pd.read_csv(config.CSV_PATH)
    return [dict(product_id=str(r.product_id), brand=str(r.brand), model=str(r.model)) for r in df.itertuples()]
