import pandas as pd


def load_data(data_path: str, text_column: str, sentiment_col: str) -> pd.DataFrame:
    """
    Load data from a text file where each line contains text and label separated by '@'.
    Lines without the delimiter are skipped.

    Args:
        data_path (str): Path to the input file.
        text_column (str): Name of the text column.
        sentiment_col (str): Name of the label column.

    Returns:
        pd.DataFrame: A DataFrame with columns [target_column, label_column].
    """
    data = []
    skipped = 0
    print(f"Loading data from: {data_path} ...")
    with open(data_path, "r", encoding="latin1") as f:
        for line in f:
            line = line.strip()
            if "@" not in line:
                skipped += 1
                continue
            parts = line.split("@", 1)  # split only on first '@'
            data.append(parts)

    print(f"Loaded {len(data)} lines.")
    print(f"Skipped {skipped} lines without labels.")
    return pd.DataFrame(data, columns=[text_column, sentiment_col])
