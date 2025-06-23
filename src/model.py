
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

def define_pipeline(preprocessor):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", SGDClassifier(random_state=42, max_iter=1000))
    ])

    param_grid = [
        {
            "preprocessor__textual_features__vectorizer__max_features": [100, 10000, 50000, 100000],
            "classifier__alpha": [0.00001, 0.0001, 0.001, 0.01],
            "classifier__penalty": ["l2", "l1"],
            "classifier__loss": ["log_loss", "hinge"]
        }
    ]
    search_cv = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    return search_cv


def model_build():
    pass