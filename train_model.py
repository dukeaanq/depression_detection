import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.catboost
import time
import pickle

def load_data(processed_dir="data/processed"):
    """
    Загружает train, valid и test датасеты из папки processed.
    """
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(processed_dir, "valid.csv"))
    test_df  = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    # Разделяем признаки и целевую переменную
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]

    X_valid = valid_df.drop("label", axis=1)
    y_valid = valid_df["label"]

    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def train_model(X_train, y_train, X_valid, y_valid, cat_cols,params):
    """
    Обучает модель CatBoost на тренировочных данных с учётом категориальных признаков.
    """
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_valid, y_valid))
    return model

def evaluate_model(model, X_test, y_test):
    """
    Оценивает модель на тестовых данных и возвращает ROC AUC.
    """
    predictions = model.predict_proba(X_test)[:, 1]  # вероятность класса 1
    auc = roc_auc_score(y_test, predictions)
    print(f"ROC AUC на тестовой выборке: {auc:.4f}")
    return auc

def save_model_pickle(model, pkl_path="models/catboost_model.pkl"):
    """
    Сохраняет модель в формате Pickle (.pkl).
    """
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Модель сохранена в {pkl_path}")

if __name__ == "__main__":
    # cat_cols нужно определить заранее, в соответствии с исходными категориальными признаками
    cat_cols = ['family_status', 'gender']

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data("data/processed")

    params = {
        "iterations": 500,
        "learning_rate": 0.01,
        "early_stopping_rounds": 20,
        "verbose": 10
    }

    mlflow.set_experiment("depression_classification_project")
    with mlflow.start_run():

        # Логируем параметры
        mlflow.log_params(params)
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("valid_rows", X_valid.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])
        mlflow.log_param("features_count", X_train.shape[1])

        start_time = time.time()
        # Обучаем модель
        model = train_model(X_train, y_train, X_valid, y_valid, cat_cols, params)
        train_duration = time.time() - start_time

        # Логируем время обучения
        mlflow.log_metric("train_duration_sec", train_duration)


        # Оцениваем модель
        auc_score = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("roc_auc_test", auc_score)



        pkl_path = "models/catboost_model.pkl"
        save_model_pickle(model, pkl_path)

        mlflow.catboost.log_model(model,
                                  name = "model",
                                  input_example = X_train.head(5))




