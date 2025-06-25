import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse
from time import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
)
import wandb
import yaml

from src import loader, preprocessing, visual
from src.utils import create_reports_subfolder, make_model_subfolder
from src.model import FeedforwardNeuralNetModel


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

    wandb.log(
        {
            f"{model_name}_{vectorizer_name}/accuracy": acc / 100,
            f"{model_name}_{vectorizer_name}/precision": precision / 100,
            f"{model_name}_{vectorizer_name}/recall": recall / 100,
            f"{model_name}_{vectorizer_name}/f1_score": f1 / 100,
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

    wandb.log({f"conf_matrix_{model_name}": wandb.Image(save_path)})


def to_dense(X):
    return X.toarray() if scipy.sparse.issparse(X) else np.array(X)


def train_model(
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
    num_epochs=50,
    lr=0.001,
    weight_decay=0.0001,
    label_smoothing=0.1,
):
    model_dir = make_model_subfolder(save_dir, model_name)
    print(f"\nTraining {model_name} ({method})...")

    X_train_tensor = torch.tensor(to_dense(X_train), dtype=torch.float32)
    X_test_tensor = torch.tensor(to_dense(X_test), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_model_state = None
    train_loss, val_loss = [], []

    for epoch in range(num_epochs):
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

        wandb.log(
            {
                f"{model_name}_{method}/train_loss": loss.item(),
                f"{model_name}_{method}/val_loss": val_loss_val.item(),
                "epoch": epoch + 1,
            }
        )

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor)
        y_preds = torch.argmax(test_preds, dim=1)

    acc = (y_preds == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"Test Accuracy: {acc:.4f}")
    report = classification_report(
        y_test.values, y_preds.cpu().numpy(), target_names=labels, zero_division=0
    )
    print(report)

    report_path = os.path.join(model_dir, f"report_{method}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    loss_plot_path = os.path.join(model_dir, f"loss_{method}.png")
    visual.plot_loss(train_loss, val_loss, save=True, save_path=loss_plot_path)

    cm = confusion_matrix(y_test.values, y_preds.cpu().numpy())
    cm_path = os.path.join(model_dir, f"conf_matrix_{method}.png")
    visual.plot_conf_matrix(
        cm=cm, labels=labels, method=method, save_path=cm_path, model_name=model_name
    )

    f1 = f1_score(y_test_tensor.cpu(), y_preds.cpu(), average="weighted")
    wandb.log({f"{model_name}_{method}/f1_score": f1})

    log_results(
        results, model_name, method, y_test.values, y_preds.cpu().numpy(), report_path
    )
    wandb.log(
        {
            "conf_matrix_nn_image": wandb.Image(cm_path),
            "loss_curve_image": wandb.Image(loss_plot_path),
        }
    )


def perform_grid_search(model, param_grid, X_train, y_train):
    grid = GridSearchCV(model, param_grid, cv=5, scoring="f1_weighted")
    grid.fit(X_train, y_train)
    print(f"Best parameters for {model.__class__.__name__}: {grid.best_params_}")
    return grid.best_estimator_


def main():
    start_time = time()

    data_path = "data/Sentences_50Agree.txt"
    text_col = "news"
    label_col = "sentiment"
    save_dir = create_reports_subfolder("training_results")
    hidden_dim = wandb.config.hidden_dim
    num_epochs = wandb.config.epochs
    lr = wandb.config.lr
    weight_decay = wandb.config.weight_decay
    label_smoothing = wandb.config.label_smoothing
    results = []

    df = loader.load_data(data_path, text_col, label_col)
    simple_cleaned_corpora, clean_column_name = (
        preprocessing.basic_clean_for_statistics(df, text_col)
    )
    visual.generate_dataset_insights(
        simple_cleaned_corpora,
        clean_column_name,
        label_col,
        save=True,
        save_dir="insights_before_cleaning",
    )

    df_clean, processed_col = preprocessing.preprocess(
        df.copy(), text_column=text_col, label_column=label_col, apply_cleaning=True
    )
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
        test_size=0.2,
        random_state=42,
        stratify=df_clean[label_col],
    )

    method = wandb.config.vectorizer
    print(f"\n\033[94m--- Vectorizer: {method} ---\033[0m")
    vectorizer = preprocessing.build_text_vectorizer(method)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

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
        LogisticRegression(max_iter=10000),
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

    input_dim = X_train_vec.shape[1]
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, len(labels))
    train_model(
        model,
        "FeedforwardNN",
        X_train_vec,
        X_test_vec,
        y_train,
        y_test,
        labels,
        method,
        save_dir,
        results,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(save_dir, "summary_results.csv"), index=False, sep=";"
    )

    best_row = results_df.loc[results_df["Weighted F1 Score"].idxmax()]
    wandb.log(
        {
            "best_model/f1_score": best_row["Weighted F1 Score"] / 100,
            "best_model/accuracy": best_row["Accuracy"] / 100,
            "best_model/name": best_row["Model"],
            "best_model/vectorizer": best_row["Vectorizer"],
        }
    )

    print(f"\nSaved summary to: {os.path.join(save_dir, 'summary_results.csv')}")
    print(f"\n\033[92mTotal runtime: {(time() - start_time)/60:.2f} minutes\033[0m")


def sweep_main():
    wandb.init()
    main()


if __name__ == "__main__":
    sweep_main()
