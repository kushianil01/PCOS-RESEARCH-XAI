"""Single, frozen, stratified train/test split. Test rows are never used for tuning, feature-set choice or thresholds."""
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config as C


def make_split(df: pd.DataFrame):
    tr, te = train_test_split(df["row_id"], test_size=C.TEST_SIZE, stratify=df[C.TARGET], random_state=C.SEED)
    return sorted(tr.tolist()), sorted(te.tolist())


def save_split(df: pd.DataFrame) -> dict:
    tr, te = make_split(df)
    d = {"seed": C.SEED, "test_size": C.TEST_SIZE, "stratified_on": C.TARGET, "train_row_ids": tr, "test_row_ids": te,
         "n_train": len(tr), "n_test": len(te),
         "train_pos": int(df[df.row_id.isin(tr)][C.TARGET].sum()), "test_pos": int(df[df.row_id.isin(te)][C.TARGET].sum())}
    C.SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    C.SPLIT_FILE.write_text(json.dumps(d))
    return d


def load_split(df: pd.DataFrame):
    d = json.loads(C.SPLIT_FILE.read_text())
    train = df[df.row_id.isin(d["train_row_ids"])].reset_index(drop=True)
    test = df[df.row_id.isin(d["test_row_ids"])].reset_index(drop=True)
    return train, test, d
