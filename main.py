from src import loader, preprocessing, visual
from src.model import FeedforwardNeuralNetModel
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from src.utils import create_reports_subfolder
from time import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

DATA_PATH = "data/Sentences_50Agree.txt"
text_column = "news"
label_column = "sentiment"
save_dir = create_reports_subfolder("training_results")


def train_feedforwardNeuralNetModel(X_train, X_test, y_train, y_test, labels, method):
    print("Training the feedforward neural net model...")

    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32) #Convert the sparse matrix to a NumPy array
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long) #Convert the Series to a NumPy array
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32) #Convert the sparse matrix to a NumPy array
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long) #Convert the Series to a NumPy array

    input_dim = X_train_tensor.shape[1]
    hidden_dim = 1000
    output_dim = len(labels)
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 100

    train_loss = []
    val_loss = []
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()


        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)
            loss_validation = criterion(predictions, y_test_tensor)
            val_loss.append(loss_validation.item())
            predicted_classes = torch.argmax(predictions, dim=1)
            correct = (predicted_classes == y_test_tensor).sum().item()
            accuracy = correct / y_test_tensor.size(0)
            print(f"Training Loss: {loss.item():.4f}, Validation Loss: {loss_validation.item():.4f}, Accuracy: {accuracy:.4f}")
            # Save best model
            if loss_validation.item() < best_val_loss:
                print("\033[92mSaving model...\033[0m")
                best_val_loss = loss_validation.item()
                best_model_state = model.state_dict()

    # Load the best model
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        y_preds = torch.argmax(predictions, dim=1)
        correct = (y_preds == y_test_tensor).sum().item()
        accuracy = correct / y_test_tensor.size(0)
        print(f"\033[92mTest Accuracy: {accuracy:.4f}\033[0m")

    visual.plot_loss(train_loss, val_loss, save=True, save_path=f"{save_dir}/loss_curve_nn_model_{method}_vectorizer.png")
    print(classification_report(y_test.values,y_preds.cpu().numpy(), target_names=labels, zero_division=0))

    #Confusion Matrix
    cm = confusion_matrix(y_test.values, y_preds.cpu().numpy())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix_nn_model_{method}_vectorizer.png")
    #plt.show()



def main():
    start_time = time()
    corpora = loader.load_data(DATA_PATH, text_column, label_column)

    # Applies simple preprocessing to the text column for statistical analysis
    simple_cleaned_corpora, clean_column_name = preprocessing.basic_clean_for_statistics(corpora, text_column)

    # Generate basic statistics and visualizations
    visual.generate_dataset_insights(simple_cleaned_corpora, clean_column_name, label_column, save=True, save_dir="insights_before_cleaning")

    # preprocessing to the text column
    df_clean, processed_col = preprocessing.preprocess(df=corpora.copy(), text_column=text_column, label_column=label_column, apply_cleaning=True)

    # Generate statistics and visualizations
    visual.generate_dataset_insights(df_clean, processed_col, label_column, save=True, save_dir="insights_after_cleaning")

    label_encoder = LabelEncoder()
    df_clean[label_column] = label_encoder.fit_transform(df_clean[label_column])
    #df_clean[label_column] = preprocessing.encode_sentiment_labels(df_clean[label_column].astype(str).str.lower())
    X_train, X_test, y_train, y_test = train_test_split(df_clean[processed_col],df_clean[label_column],test_size=0.3,random_state=42,stratify=df_clean[label_column])


    #ToDo: loop over all vectorizer methods
    methods = ['count', 'binary', 'tfidf', 'hashing' ]
    for method in methods:
        print(f"Training model using {method} vectorizer...")
        vectorizer = preprocessing.build_text_vectorizer(method=method)
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
        plt.savefig(f"{save_dir}/ConfusionMatrix_Naive Bayes_and_Logistic Regression_{method}_vectorizer.png")
        #plt.show()

        train_feedforwardNeuralNetModel(X_train_vec, X_test_vec, y_train, y_test, label_encoder.classes_, method)
    end_time = time()
    print(f"\033[92mTotal time: {(end_time - start_time)/60:.2f}\033[0m")


if __name__ == "__main__":
    main()


