import os
import pandas as pd


def count_number_of_words(df: pd.DataFrame, column_to_count: str, count_to: str = "number_of_words") -> pd.DataFrame:
    df[count_to] = df[column_to_count].apply(lambda x: len(x.split()))
    return df

def get_unique_words(texts):
    """
    Return a set of unique words from a list/series of texts.
    """
    all_words = ' '.join(texts).split()
    return set(all_words)



def create_reports_subfolder(subfolder_name: str) -> str:
    """Creates a subfolder inside the 'reports' directory located one level above the current directory."""
    reports_path = os.path.join(".", "reports", subfolder_name)
    os.makedirs(reports_path, exist_ok=True)
    return reports_path

def print_sentiment_distribution(df: pd.DataFrame, sentiment_col: str):
    """Prints the sentiment distribution with count (percentage)."""
    counts = df[sentiment_col].value_counts()
    total = counts.sum()

    print("Sentiment Distribution")

    for sentiment in counts.index:
        count = counts[sentiment]
        pct = count / total * 100
        print(f"{sentiment:<10} {count:>5} ({pct:>5.2f}%)")
