import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Cargar el modelo de spaCy para el idioma inglés
nlp = spacy.load('en_core_web_sm')

# Función para limpiar el texto
def clean_text(text):
    # Procesar el texto con spaCy
    doc = nlp(text.lower())  # Convertir a minúsculas y procesar el texto

    # Filtrar tokens: eliminar stopwords y puntuación
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]

    # Unir los tokens filtrados de nuevo en un solo string
    return " ".join(tokens)

def main():
    # Ruta absoluta al archivo spam.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Encuentra el directorio del script
    csv_path = os.path.join(current_dir, '..', 'data', 'spam.csv')  # Sube un nivel y accede a data/spam.csv
    csv_path = os.path.normpath(csv_path)  # Normaliza la ruta (por si hay cosas como "..")

    # Leer el CSV
    df = pd.read_csv(csv_path, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']

    # Limpiar el texto
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Mapear las etiquetas 'ham' y 'spam' a 0 y 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42
    )

    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Guardar los artefactos
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(X_train_vec, 'artifacts/X_train.pkl')
    joblib.dump(X_test_vec, 'artifacts/X_test.pkl')
    joblib.dump(y_train, 'artifacts/y_train.pkl')
    joblib.dump(y_test, 'artifacts/y_test.pkl')
    joblib.dump(vectorizer, 'artifacts/vectorizer.pkl')

if __name__ == '__main__':
    main()
