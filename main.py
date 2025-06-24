from src import loader, preprocessing, visual
from src.utils import create_reports_subfolder, make_model_subfolder
from src.model import FeedforwardNeuralNetModel
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from time import time
import os


def log_results(results_list, model_name, vectorizer_name, y_true, y_pred, report_path):
    acc = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    precision *= 100
    recall *= 100
    f1 *= 100

    results_list.append(
        {
            "Model": model_name,
            "Vectorizer": vectorizer_name,
            "Accuracy": round(acc, 2),
            "Weighted Precision": round(precision, 2),
            "Weighted Recall": round(recall, 2),
            "Weighted F1 Score": round(f1, 2),
            "Report Path": report_path,
        }
    )


def train_and_evaluate_classic_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    labels,
    method,
    save_dir,
    results,
):
    model_dir = make_model_subfolder(save_dir, model_name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"\n{model_name} ({method}) Classification Report:")
    report = classification_report(y_test, preds)
    print(report)

    report_path = os.path.join(model_dir, f"report_{method}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, preds)
    save_path = os.path.join(model_dir, f"conf_matrix_{method}.png")
    visual.plot_conf_matrix(
        cm=cm, labels=labels, method=method, save_path=save_path, model_name=model_name
    )
    log_results(results, model_name, method, y_test, preds, report_path)


def train_feedforward_nn(
    X_train, X_test, y_train, y_test, labels, method, save_dir, results
):
    model_name = "FeedforwardNN"
    model_dir = make_model_subfolder(save_dir, model_name)
    print(f"\nTraining {model_name} ({method})...")

    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    input_dim = X_train_tensor.shape[1]
    model = FeedforwardNeuralNetModel(
        input_dim, hidden_dim=1000, output_dim=len(labels)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_model_state = None
    train_loss, val_loss = [], []

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss_val = criterion(val_outputs, y_test_tensor)
            val_loss.append(val_loss_val.item())
            if val_loss_val.item() < best_val_loss:
                best_val_loss = val_loss_val.item()
                best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor)
        y_preds = torch.argmax(test_preds, dim=1)

    accuracy = (y_preds == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"Test Accuracy: {accuracy:.4f}")
    report = classification_report(
        y_test.values, y_preds.cpu().numpy(), target_names=labels, zero_division=0
    )
    print(report)

    report_path = os.path.join(model_dir, f"report_{method}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    visual.plot_loss(
        train_loss,
        val_loss,
        save=True,
        save_path=os.path.join(model_dir, f"loss_{method}.png"),
    )
    cm = confusion_matrix(y_test.values, y_preds.cpu().numpy())
    save_path = os.path.join(model_dir, f"conf_matrix_{method}.png")
    visual.plot_conf_matrix(
        cm=cm, labels=labels, method=method, save_path=save_path, model_name=model_name
    )
    log_results(
        results, model_name, method, y_test.values, y_preds.cpu().numpy(), report_path
    )


def main():
    start_time = time()

    data_path = "data/Sentences_50Agree.txt"
    text_col = "news"
    label_col = "sentiment"
    save_dir = create_reports_subfolder("training_results")
    results = []

    df = loader.load_data(data_path, text_col, label_col)

    # Applies simple preprocessing to the text column for statistical analysis
    simple_cleaned_corpora, clean_column_name = (
        preprocessing.basic_clean_for_statistics(df, text_col)
    )

    # Generate basic statistics and visualizations
    visual.generate_dataset_insights(
        simple_cleaned_corpora,
        clean_column_name,
        label_col,
        save=True,
        save_dir="insights_before_cleaning",
    )

    # preprocessing to the text column
    df_clean, processed_col = preprocessing.preprocess(
        df=df.copy(), text_column=text_col, label_column=label_col, apply_cleaning=True
    )

    # Generate statistics and visualizations
    visual.generate_dataset_insights(
        df_clean,
        processed_col,
        label_col,
        save=True,
        save_dir="insights_after_cleaning",
    )

    label_encoder = LabelEncoder()
    df_clean[label_col] = label_encoder.fit_transform(df_clean[label_col])
    labels = label_encoder.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        df_clean[processed_col],
        df_clean[label_col],
        test_size=0.3,
        random_state=42,
        stratify=df_clean[label_col],
    )

    vectorizer_methods = ["count", "binary", "tfidf", "hashing"]
    for method in vectorizer_methods:
        print(f"\n\033[94m--- Vectorizer: {method} ---\033[0m")
        vectorizer = preprocessing.build_text_vectorizer(method)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train and evaluate models
        train_and_evaluate_classic_model(
            MultinomialNB(),
            "NaiveBayes",
            X_train_vec,
            X_test_vec,
            y_train,
            y_test,
            labels,
            method,
            save_dir,
            results,
        )
        train_and_evaluate_classic_model(
            LogisticRegression(max_iter=1000),
            "LogisticRegression",
            X_train_vec,
            X_test_vec,
            y_train,
            y_test,
            labels,
            method,
            save_dir,
            results,
        )
        train_feedforward_nn(
            X_train_vec, X_test_vec, y_train, y_test, labels, method, save_dir, results
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(save_dir, "summary_results.csv"), index=False, sep=";"
    )
    print(f"\nSaved summary to: {os.path.join(save_dir, 'summary_results.csv')}")
    print(f"\n\033[92mTotal runtime: {(time() - start_time)/60:.2f} minutes\033[0m")


if __name__ == "__main__":
    main()
