import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from wordcloud import WordCloud
from collections import Counter

#consistent color mapping by label name
color_map = {
    'positive': 'green',
    'negative': 'red',
    'neutral': 'gray'
}


def data_distribution_per_class(df):
    """
    Plot the distribution of data across different classes.
    """
    counts = df['label'].value_counts()
    colors = [color_map.get(label, 'gray') for label in counts.index]

    counts.plot.bar(xlabel="Label", ylabel="Count", color=colors, title='News Class Distribution')
    plt.show()


def get_unique_words(texts):
    """
    Return a set of unique words from a list/series of texts.
    """
    all_words = ' '.join(texts).split()
    return set(all_words)


def unique_words_per_class(df, text_column):
    """
    Plot number of unique words per class and draw a line for total unique words across all classes.
    """
    labels = df['label'].unique()
    unique_counts = {label: len(get_unique_words(df[df['label'] == label][text_column])) for label in labels}

    for label, count in unique_counts.items():
        print(f"Unique words in class {label}: {count}")

    total_unique = len(get_unique_words(df[text_column]))
    print("Total unique words:", total_unique)

    colors = [color_map.get(label, 'gray') for label in unique_counts.keys()]

    plt.figure(figsize=(8, 5))
    plt.bar(unique_counts.keys(), unique_counts.values(), color=colors)
    plt.axhline(y=total_unique, color='black', linestyle='--')


    legend_elements = [Patch(facecolor=color_map.get(label, 'gray'), label=f'{label}') for label in
                       unique_counts.keys()]
    legend_elements.append(Patch(facecolor='black', label='Total unique'))

    plt.legend(handles=legend_elements)
    plt.title("Unique Word Counts per Class (with Total Line)")
    plt.ylabel("Unique Word Count")
    plt.xlabel("Class")
    plt.show()


def most_common_words_per_class(df, text_column, top_n=30):
    """
    Plot the most common words in each class as a bar chart.
    """
    labels = df['label'].unique()
    for label in labels:
        words = ' '.join(df[df['label'] == label][text_column]).split()
        common = Counter(words).most_common(top_n)
        words_, counts = zip(*common)

        color = color_map.get(label, 'gray')
        plt.figure(figsize=(10, 5))
        plt.bar(words_, counts, color=color)
        plt.title(f"Top {top_n} Words in Class {label}")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def word_cloud_per_class(df, text_column):
    """
    Generate and display a word cloud for each class.
    """
    labels = df['label'].unique()
    for label in labels:
        text = ' '.join(df[df['label'] == label][text_column])
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f'Word Cloud - Class {label}')
        plt.axis('off')
        plt.show()


def avg_word_count_per_class(df, column):
    """
    Plot the average word count per class using the specified column.
    """
    group_avg = df.groupby('label')[column].mean()
    colors = [color_map.get(label, 'gray') for label in group_avg.index]
    group_avg.plot.bar(color=colors, title='Average Word Count per Class')
    plt.ylabel('Average Word Count')
    plt.xlabel('Class')
    plt.show()
