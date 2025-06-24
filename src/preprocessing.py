import pandas as pd
import numpy as np
import spacy
import string
from typing import List, Tuple
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    HashingVectorizer,
)
from sklearn.base import TransformerMixin
from gensim.downloader import load as gensim_load
from sklearn.model_selection import train_test_split
from src import visual

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# ---------------- Utility Functions ---------------- #


def clean_text(text) -> str:
    """Lowercases text, removes punctuation and stopwords."""
    stop_words = set(stopwords.words("english"))
    punct_table = str.maketrans("", "", string.punctuation)

    if not isinstance(text, str):
        return ""
    text = text.lower().translate(punct_table)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def remove_empty_or_whitespace_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Removes rows where the given column is empty, whitespace, or invalid content.
    """
    return df[
        df[column].notna()
        & ~df[column].str.strip().eq("")
        & ~df[column].str.strip().eq("` `")
    ]


def lemmatize_and_clean_text(doc) -> str:
    """Lemmatizes a spaCy Doc and removes stopwords and punctuation."""
    return " ".join(
        token.lemma_ for token in doc
    )  # if not token.is_stop and not token.is_punct


def remove_duplicates(
    df: pd.DataFrame, subset: List[str], keep: str = "first"
) -> pd.DataFrame:
    """Remove duplicate rows based on specified columns."""
    return df.drop_duplicates(subset=subset, keep=keep)


def encode_sentiment_labels(labels: pd.Series) -> pd.Series:
    """
    Converts sentiment labels into numerical values.
    'positive' -> 1, 'neutral' -> 0, 'negative' -> -1
    """
    mapping = {"positive": 1, "neutral": 0, "negative": -1}
    return labels.map(mapping).astype(int)


# ---------------- Text Processing Functions ---------------- #


def basic_clean_for_statistics(
    df: pd.DataFrame, text_column: str
) -> Tuple[pd.DataFrame, str]:
    """
    Applies simple preprocessing to the target column for statistical analysis:
    - Lowercase
    - Remove punctuation
    - Remove stopwords

       The cleaned text is stored in a new column named 'clean_<target_column>'.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Column to clean.

    Returns:
        Tuple: (cleaned DataFrame, name of cleaned text column)
    """
    cleaned_df = df.copy()
    clean_column_name = f"clean_{text_column}"
    cleaned_df[clean_column_name] = cleaned_df[text_column].apply(clean_text)
    return cleaned_df, clean_column_name


def preprocess_text_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, str]:
    """
    Preprocesses a text column:
    - Removes empty/whitespace rows
    - Lowercases and strips text
    - Lemmatizes using spaCy
    - Removes stopwords and punctuation
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    original_rows = len(df)
    df_clean = remove_empty_or_whitespace_rows(df, column)

    text_list = df_clean[column].str.lower().str.strip().tolist()
    docs = nlp.pipe(text_list)
    new_column = f"cleaned_{column}"
    df_clean[new_column] = [lemmatize_and_clean_text(doc) for doc in docs]

    df_clean = remove_empty_or_whitespace_rows(df_clean, new_column)

    print(f"Text Preprocessing Complete:")
    print(f"- Original rows: {original_rows}")
    print(f"- Rows after removing empty text: {len(df_clean)}")
    print(f"- New column created: '{new_column}'\n")

    return df_clean, new_column


def preprocess(
    df: pd.DataFrame, text_column: str, label_column: str, apply_cleaning: bool = True
) -> Tuple[pd.DataFrame, str]:
    """
    Preprocesses the text column by cleaning and removing duplicates.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Column with raw text.
        label_column (str): Column with sentiment labels.
        apply_cleaning (bool): If True, applies full text preprocessing.

    Returns:
        Tuple[pd.DataFrame, str]: Cleaned DataFrame and name of processed text column.
    """
    original_rows = len(df)
    cleaned_df = df.copy()
    # Duplicate check before cleaning
    dupes_before = (
        cleaned_df.groupby([text_column, label_column]).size().reset_index(name="count")
    )
    num_dupes_before = len(dupes_before.query("count > 1"))

    if apply_cleaning:
        cleaned_df, processed_col = preprocess_text_column(cleaned_df, text_column)

        # Duplicate check after cleaning
        dupes_after = (
            cleaned_df.groupby([processed_col, label_column])
            .size()
            .reset_index(name="count")
        )
        dupes_found = dupes_after.query("count > 1")
        num_dupes_after = len(dupes_found)

        if not dupes_found.empty:
            cleaned_df = remove_duplicates(
                cleaned_df, [processed_col, label_column], keep="first"
            )

        visual.plot_most_common_words_by_sentiment(
            cleaned_df, processed_col, label_column
        )
        visual.generate_word_cloud_by_sentiment(cleaned_df, processed_col, label_column)

    else:
        processed_col = text_column
        num_dupes_after = num_dupes_before

    # Final Summary
    print("Preprocessing Pipeline Summary")
    print("------------------------------")
    print(f"Initial row count: {original_rows}")
    print(f"Duplicates before cleaning: {num_dupes_before}")
    if apply_cleaning:
        print(f"Duplicates after cleaning:  {num_dupes_after}")
        print(f"Final row count after cleaning and deduplication: {len(cleaned_df)}")
    else:
        print("Cleaning skipped.")
        print(f"Final row count after deduplication: {len(cleaned_df)}")
    print(f"Processed text column: '{processed_col}'")
    print(f"Label encoding applied: '{label_column}'\n")

    return cleaned_df, processed_col


# ---------------- Feature Extraction ---------------- #


class GloVeVectorizer(TransformerMixin):
    def __init__(self, model_name="glove-wiki-gigaword-100"):
        self.model_name = model_name
        self.model = gensim_load(model_name)
        self.dim = self.model.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(
            [
                np.mean(
                    [self.model[word] for word in doc.split() if word in self.model]
                    or [np.zeros(self.dim)],
                    axis=0,
                )
                for doc in X
            ]
        )


def build_text_vectorizer(method: str = "tfidf"):
    """
    Supported methods:
        - 'count'                      → CountVectorizer
        - 'binary'                     → CountVectorizer(binary=True)
        - 'tfidf'                      → TfidfVectorizer (1-2 grams)
        - 'hashing'                    → HashingVectorizer
        - 'glove-50'                  → GloVe 50d
        - 'glove-100' (default)       → GloVe 100d
        - 'glove-200'                 → GloVe 200d
    """
    if method == "count":
        return CountVectorizer()
    elif method == "binary":
        return CountVectorizer(binary=True)
    elif method == "tfidf":
        return TfidfVectorizer(ngram_range=(1, 3))
    elif method == "hashing":
        return HashingVectorizer(n_features=1000, alternate_sign=False)
    elif method == "glove-50":
        return GloVeVectorizer("glove-wiki-gigaword-50")
    elif method == "glove-100":
        return GloVeVectorizer("glove-wiki-gigaword-100")
    elif method == "glove-200":
        return GloVeVectorizer("glove-wiki-gigaword-200")
    else:
        raise ValueError(f"Unknown vectorization method: '{method}'")


# ---------------- Main  ---------------- #
def run_pipeline_process_and_vectorize_data(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    vectorizer_method: str = "tfidf",
    apply_cleaning: bool = True,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple:
    """
    Full pipeline: preprocess text, encode labels, split data, and vectorize text.

    Args:
        df (pd.DataFrame): Input DataFrame with text and labels.
        text_column (str): Column containing the raw text.
        label_column (str): Column containing sentiment labels.
        vectorizer_method (str): One of 'tfidf', 'count', 'binary', or 'hashing'.
        apply_cleaning (bool): Whether to apply spaCy-based text preprocessing.
        test_size (float): Proportion of data to use for test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple:
            - X_train_vec: Vectorized training features
            - y_train: Encoded training labels
            - X_test_vec: Vectorized test features
            - y_test: Encoded test labels
    """
    # Clean the text
    df_clean, processed_col = preprocess(
        df.copy(), text_column, label_column, apply_cleaning
    )

    # Encode labels
    df_clean[label_column] = encode_sentiment_labels(
        df_clean[label_column].astype(str).str.lower()
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean[processed_col],
        df_clean[label_column],
        test_size=test_size,
        random_state=random_state,
        stratify=df_clean[label_column],
    )

    # Vectorize text
    vectorizer = build_text_vectorizer(vectorizer_method)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, y_train, X_test_vec, y_test
