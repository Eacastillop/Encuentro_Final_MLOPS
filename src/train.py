# src/train.py

import spacy
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Cargar el modelo de spacy
nlp = spacy.load('en_core_web_sm')

# Cargar los datos preprocesados
def load_data():
    X_train = joblib.load('artifacts/X_train.pkl')
    X_test = joblib.load('artifacts/X_test.pkl')
    y_train = joblib.load('artifacts/y_train.pkl')
    y_test = joblib.load('artifacts/y_test.pkl')
    vectorizer = joblib.load('artifacts/vectorizer.pkl')
    return X_train, X_test, y_train, y_test, vectorizer

# Entrenamiento del modelo
def train_model(X_train, y_train):
    # Usamos Random Forest como clasificador, pero puedes cambiarlo por otro
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluar el modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Guardar el modelo entrenado
def save_model(model):
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, 'artifacts/model.pkl')

# Guardar los artefactos
def save_artifacts(vectorizer, model):
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(vectorizer, 'artifacts/vectorizer.pkl')
    joblib.dump(model, 'artifacts/model.pkl')

# Main
def main():
    # Cargar los datos
    X_train, X_test, y_train, y_test, vectorizer = load_data()

    # Entrenar el modelo
    model = train_model(X_train, y_train)

    # Evaluar el modelo
    evaluate_model(model, X_test, y_test)

    # Guardar el modelo y los artefactos
    save_artifacts(vectorizer, model)

if __name__ == '__main__':
    main()
