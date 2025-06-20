import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(doc):
    """
    Cleans a spaCy Doc object by lemmatizing and removing stopwords and punctuation.

    Args:
        doc (spacy.tokens.Doc): A spaCy Doc object.

    Returns:
        str: A cleaned string of lemmatized words.
    """
    return ' '.join(
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct
    )

def preprocess_text(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Preprocesses text data in a DataFrame column:
    - Removes rows where the target column is null, empty, or whitespace-only
    - Converts text to lowercase and strips spaces
    - Processes text with spaCy
    - Cleans text using `clean_text` which removes stopwords and punctuation.
    - Adds a new column 'new_<target_column>' with processed text

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the text column to preprocess.

    Returns:
        pd.DataFrame: A cleaned DataFrame with a new column containing cleaned text.
    """
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame.")

    original_rows = len(df)

    # Remove rows with NaN, empty, or whitespace-only values
    df_clean = df.copy()
    df_clean = df_clean[df_clean[target_column].notna()]
    df_clean = df_clean[df_clean[target_column].str.strip() != '']

    cleaned_rows = len(df_clean)

    # Preprocess text
    corpus = df_clean[target_column].str.lower().str.strip().tolist()
    docs = nlp.pipe(corpus)
    new_col_name = f'new_{target_column}'
    df_clean[new_col_name] = [clean_text(doc) for doc in docs]

    print("[Text Preprocessing Summary]")
    print(f"- Original rows: {original_rows}")
    print(f"- Removed rows (null/empty): {original_rows - cleaned_rows}")
    print(f"- Processed column: '{target_column}'")
    print(f"- New column added: '{new_col_name}'")
    print(f"- Final row count: {cleaned_rows}")
    print("Preprocessing complete.\n")

    return df_clean

def remove_duplicates(df: pd.DataFrame, target_columns: list[str], keep: str = 'first') -> pd.DataFrame:
    """
    Removes duplicate rows based on specified columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_columns (list[str]): Columns to consider for identifying duplicates.
        keep (str): Which duplicate to keep ('first', 'last', or False).

    Returns:
        pd.DataFrame: A DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=target_columns, keep=keep)

def clean_transform():
    pass