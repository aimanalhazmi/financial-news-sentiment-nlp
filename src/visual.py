import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from wordcloud import WordCloud
from collections import Counter
import matplotlib.gridspec as gridspec
from src.utils import (
    count_number_of_words,
    get_unique_words,
    create_reports_subfolder,
    print_sentiment_distribution,
)

# consistent color mapping by label name
color_map = {
    "positive": "green",
    1: "green",
    "negative": "red",
    -1: "red",
    "neutral": "gray",
    0: "gray",
}


def plot_word_count_distribution_by_sentiment(
    df, text_length_col, sentiment_col, save=False, save_path=""
):
    """
    Plot the distribution of word counts across different sentiment classes.
    """
    unique_sentiments = df[sentiment_col].unique()
    for sentiment in unique_sentiments:
        sentiment_subset = df[df[sentiment_col] == sentiment]
        plt.hist(
            sentiment_subset[text_length_col], alpha=0.5, label=str(sentiment), bins=15
        )
    plt.legend()
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.title("Text Length Distribution Across Sentiments")
    if save and save_path:
        plt.savefig(save_path, bbox_inches="tight")
    #    if not save:
    #        plt.show()
    plt.close()


def plot_avg_word_count_by_sentiment(
    df, text_length_col, sentiment_col, save=False, save_path=""
):
    """
    Plot the average text length (in words) for each sentiment class.
    """
    group_avg = df.groupby(sentiment_col)[text_length_col].mean()
    colors = [color_map.get(label, "gray") for label in group_avg.index]
    group_avg.plot.bar(color=colors, title="Average Text Length by Sentiment")
    plt.ylabel("Average Text Length (Number of Words)")
    plt.xlabel("Sentiment")
    if save and save_path:
        plt.savefig(save_path, bbox_inches="tight")
    #    if not save:
    #        plt.show()
    plt.close()


def plot_data_distribution_by_sentiment(df, sentiment_col, save=False, save_path=""):
    """
    Plot the distribution of data across different sentiment classes.
    """
    counts = df[sentiment_col].value_counts()
    colors = [color_map.get(label, "gray") for label in counts.index]

    counts.plot.bar(
        xlabel="Sentiment",
        ylabel="Count",
        color=colors,
        title="Data Distribution by Sentiment",
    )
    if save and save_path:
        plt.savefig(save_path, bbox_inches="tight")
    #    if not save:
    #        plt.show()
    plt.close()


def plot_unique_words_by_sentiment(
    df, text_column, sentiment_col, save=False, save_path=""
):
    """
    Plot the number of unique words used in texts for each sentiment class,
    and draw a horizontal line representing the total number of unique words across all classes.
    """
    sentiments = df[sentiment_col].unique()
    unique_counts = {
        sentiment: len(
            get_unique_words(df[df[sentiment_col] == sentiment][text_column])
        )
        for sentiment in sentiments
    }

    for sentiment, count in unique_counts.items():
        print(f"Unique words in sentiment '{sentiment}': {count}")

    total_unique = len(get_unique_words(df[text_column]))
    print("Total unique words across all sentiments:", total_unique)

    colors = [color_map.get(sentiment, "gray") for sentiment in unique_counts.keys()]

    plt.figure(figsize=(8, 5))
    plt.bar(unique_counts.keys(), unique_counts.values(), color=colors)
    plt.axhline(y=total_unique, color="black", linestyle="--")

    legend_elements = [
        Patch(facecolor=color_map.get(sentiment, "gray"), label=f"{sentiment}")
        for sentiment in unique_counts.keys()
    ]
    legend_elements.append(Patch(facecolor="black", label="Total Unique Words"))

    plt.legend(handles=legend_elements)
    plt.title("Unique Word Counts by Sentiment Class")
    plt.ylabel("Unique Word Count")
    plt.xlabel("Sentiment")
    if save and save_path:
        plt.savefig(save_path, bbox_inches="tight")
    #    if not save:
    #        plt.show()
    plt.close()


def plot_most_common_words_by_sentiment(
    df, text_column, sentiment_col, top_n=30, save=False, save_path=""
):
    """
    Plot the most common words in each sentiment class as a bar chart.
    """
    sentiments = df[sentiment_col].unique()
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    axes = []
    for i in range(len(sentiments)):
        if i < 2:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[1, :])
        axes.append(ax)

    for i, sentiment in enumerate(sentiments):
        words = " ".join(df[df[sentiment_col] == sentiment][text_column]).split()
        common = Counter(words).most_common(top_n)
        words_, counts = zip(*common)

        color = color_map.get(sentiment, "gray")
        axes[i].bar(words_, counts, color=color)
        axes[i].set_title(
            f"Top {top_n} Words in {sentiment.capitalize()} Financial News", fontsize=24
        )
        axes[i].set_xlabel("Words")
        axes[i].set_ylabel("Frequency")
        axes[i].tick_params(axis="x", labelrotation=60)

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path, bbox_inches="tight")
    #    if not save:
    #        plt.show()
    plt.close()


def generate_word_cloud_by_sentiment(
    df, text_column, sentiment_col, save=False, save_path=""
):
    """
    Generate and display a word cloud for each sentiment class.
    """
    sentiments = df[sentiment_col].unique()
    fig, ax = plt.subplots(1, len(sentiments), figsize=(9 * len(sentiments), 12))
    for i, sentiment in enumerate(sentiments):
        text = " ".join(df[df[sentiment_col] == sentiment][text_column])
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        ax[i].imshow(wc, interpolation="bilinear")
        ax[i].set_title(f"{sentiment.capitalize()} Financial News", fontsize=24)
        ax[i].axis("off")
    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path, bbox_inches="tight")
    #    if not save:
    #        plt.show()
    plt.close()


def plot_loss(train_loss, val_loss, save, save_path):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_conf_matrix(cm, labels, method, save_path="conf_matrix.png", model_name=None):
    """Plots and saves a heatmap of the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    title = (
        f"Confusion Matrix - {method}"
        if not model_name
        else f"{model_name} Confusion Matrix ({method})"
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def generate_dataset_insights(
    df: pd.DataFrame,
    text_column: str,
    sentiment_col: str,
    save: bool = False,
    save_dir: str = "insights",
):
    """
    Generates basic statistics and visualizations for a sentiment-labeled text dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Column containing the text.
        sentiment_col (str): Column containing sentiment labels.
        save (bool): If True, saves plots to the specified directory.
        save_dir (str): Directory to save plots.
    """

    save_dir = create_reports_subfolder(save_dir)
    print(f"Generating dataset insights for {text_column} in {save_dir}..")

    # Add a word count column
    df = count_number_of_words(
        df=df, column_to_count=text_column, count_to="word_count"
    )

    # Print basic statistics
    print("Dataset Information")
    print("-------------------")
    print(f"Total samples: {len(df)}")
    print_sentiment_distribution(df=df, sentiment_col=sentiment_col)
    print(f"Average Text Length: {df['word_count'].mean():.2f}\n")

    # Generate all plots
    plot_data_distribution_by_sentiment(
        df,
        sentiment_col=sentiment_col,
        save=save,
        save_path=f"{save_dir}/sentiment_distribution.png",
    )

    plot_word_count_distribution_by_sentiment(
        df,
        text_length_col="word_count",
        sentiment_col=sentiment_col,
        save=save,
        save_path=f"{save_dir}/word_count_distribution.png",
    )

    plot_avg_word_count_by_sentiment(
        df,
        text_length_col="word_count",
        sentiment_col=sentiment_col,
        save=save,
        save_path=f"{save_dir}/avg_word_count.png",
    )

    plot_unique_words_by_sentiment(
        df,
        text_column=text_column,
        sentiment_col=sentiment_col,
        save=save,
        save_path=f"{save_dir}/unique_words.png",
    )

    plot_most_common_words_by_sentiment(
        df,
        text_column=text_column,
        sentiment_col=sentiment_col,
        save=save,
        save_path=f"{save_dir}/most_common_words.png",
    )

    generate_word_cloud_by_sentiment(
        df,
        text_column=text_column,
        sentiment_col=sentiment_col,
        save=save,
        save_path=f"{save_dir}/word_cloud.png",
    )

    print("Insight generation complete.\n")
