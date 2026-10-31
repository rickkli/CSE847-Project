# Modified by Cursor — shared feature engineering and augmented LR pipeline for CSE847 notebooks.

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLS = [
    "prompt_text",
    "response_text",
    "response_clean",
    "prompt_type",
    "task_type",
]


def light_clean(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text


def tokenize(text):
    return light_clean(text).split()


HEDGE_WORDS = {
    "may", "might", "could", "possibly", "perhaps",
    "likely", "generally", "often", "sometimes",
}

CERTAINTY_WORDS = {
    "always", "never", "definitely", "certainly", "clearly",
    "must", "impossible", "every", "all", "none",
}


def rate(word_set, tokens):
    n = len(tokens)
    if n == 0:
        return 0.0
    return sum(t in word_set for t in tokens) / n


def type_jaccard(prompt_tokens, response_tokens):
    p, r = set(prompt_tokens), set(response_tokens)
    if not p and not r:
        return 0.0
    return len(p & r) / len(p | r)


def unique_token_rate(prompt_tokens, response_tokens):
    r = set(response_tokens)
    if not r:
        return 0.0
    p = set(prompt_tokens)
    return len(r - p) / len(r)


def handcrafted_row(prompt_text, response_text):
    pt = tokenize(prompt_text)
    rt = tokenize(response_text)
    prompt_n = len(pt)
    response_n = len(rt)
    ratio = response_n / max(prompt_n, 1)
    return [
        response_n,
        ratio,
        type_jaccard(pt, rt),
        unique_token_rate(pt, rt),
        rate(HEDGE_WORDS, rt),
        rate(CERTAINTY_WORDS, rt),
    ]


HANDCRAFTED_NAMES = [
    "response_word_count",
    "response_to_prompt_word_ratio",
    "token_jaccard",
    "response_unique_token_rate",
    "hedge_rate",
    "certainty_rate",
]


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [
            handcrafted_row(p, r)
            for p, r in zip(X["prompt_text"], X["response_text"])
        ]
        return np.asarray(rows, dtype=float)


def prepare_feature_frame(row_or_df):
    if isinstance(row_or_df, dict):
        df = pd.DataFrame([row_or_df])
    elif isinstance(row_or_df, pd.Series):
        df = row_or_df.to_frame().T
    else:
        df = row_or_df.copy()

    df["prompt_text"] = df["prompt_text"].astype(str)
    df["response_text"] = df["response_text"].astype(str)
    df["response_clean"] = df["response_text"].map(light_clean)

    for col in ("prompt_type", "task_type"):
        if col not in df.columns:
            df[col] = "unknown"
        else:
            df[col] = df[col].astype(str)

    return df[FEATURE_COLS]


def build_augmented_lr_pipeline():
    preprocess = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100, sublinear_tf=True), "response_clean"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["prompt_type", "task_type"]),
            ("hand", Pipeline(
                    steps=[
                        ("extract", HandcraftedFeatures()),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["prompt_text", "response_text"],
            ),
        ],
        sparse_threshold=0.3,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("lr", LogisticRegression(max_iter=1000, random_state=1, class_weight="balanced")),
        ]
    )