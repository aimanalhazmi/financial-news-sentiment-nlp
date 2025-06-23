from src import loader, model, preprocessing, visual
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

DATA_PATH = "data/Sentences_50Agree.txt"
text_column = "news"
label_column = "sentiment"


def main():
    corpora = loader.load_data(DATA_PATH, text_column, label_column)

    # Applies simple preprocessing to the text column for statistical analysis
    simple_cleaned_corpora, clean_column_name = preprocessing.basic_clean_for_statistics(corpora, text_column)

    # Generate basic statistics and visualizations
    visual.generate_dataset_insights(simple_cleaned_corpora, clean_column_name, label_column, save=True, save_dir="insights_before_cleaning")

    # preprocessing to the text column
    df_clean, processed_col = preprocessing.preprocess(df=corpora.copy(), text_column=text_column, label_column=label_column, apply_cleaning=True)

    # Generate statistics and visualizations
    visual.generate_dataset_insights(df_clean, processed_col, label_column, save=True, save_dir="insights_after_cleaning")

    df_clean[label_column] = preprocessing.encode_sentiment_labels(df_clean[label_column].astype(str).str.lower())
    X_train, X_test, y_train, y_test = train_test_split(df_clean[processed_col],df_clean[label_column],test_size=0.3,random_state=42,stratify=df_clean[label_column])

    #ToDo: loop over all  vectorizer methods
    vectorizer = preprocessing.build_text_vectorizer(method='count')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)

    # Predictions and evaluation
    nb_preds = nb_model.predict(X_test_vec)

    print("Naive Bayes Classification Report:")
    print(classification_report(y_test, nb_preds))

    lr_model = LogisticRegression()
    lr_model.fit(X_train_vec, y_train)

    # Predictions and evaluation
    lr_preds = lr_model.predict(X_test_vec)

    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, lr_preds))

    # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ConfusionMatrixDisplay.from_estimator(nb_model, X_test_vec, y_test, ax=axes[0], cmap="Blues")
    axes[0].title.set_text('Naive Bayes')

    ConfusionMatrixDisplay.from_estimator(lr_model, X_test_vec, y_test, ax=axes[1], cmap="Greens")
    axes[1].title.set_text('Logistic Regression')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


