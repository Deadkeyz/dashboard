import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Titre du dashboard
st.set_page_config(page_title="Projet Data Science", layout="wide")

# Fonction pour charger les données
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Uploader le fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

def main():
    menu = ["Accueil", "Aperçu des données", "Analyse Exploratoire des Données", "Préparation des données", "Modélisation"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Accueil":
        st.title("Projet Data Science - Prédiction du Risque de Défaut de Crédit")
        st.write("""
            Ce dashboard interactif permet de charger un fichier CSV, d'explorer les données,
            de les préparer, de construire des modèles de prédiction et d'évaluer leurs performances.
        """)

    elif choice == "Aperçu des données":
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.header("Aperçu des données")
            st.write(data.head())
        else:
            st.warning("Veuillez uploader un fichier CSV pour voir l'aperçu des données.")

    elif choice == "Analyse Exploratoire des Données":
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.header("Analyse Exploratoire des Données")

            st.subheader("Statistiques descriptives")
            st.write(data.describe())

            st.subheader("Distribution des variables (Univarié)")
            columns = data.columns
            for col in columns:
                fig = px.histogram(data, x=col, marginal="box", title=f"Distribution de {col}")
                st.plotly_chart(fig)

            st.subheader("Relations entre les variables (Bivarié)")
            target_column = st.selectbox("Sélectionnez la colonne cible pour les graphiques bivariés", data.columns)
            for col in columns:
                if data[col].dtype in ['int64', 'float64'] and col != target_column:
                    fig = px.scatter(data, x=col, y=target_column, title=f"Relation entre {col} et {target_column}", marginal_y="violin", marginal_x="box")
                    st.plotly_chart(fig)
        else:
            st.warning("Veuillez uploader un fichier CSV pour analyser les données.")

    elif choice == "Préparation des données":
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.header("Préparation des données")
            st.write("Vérification des valeurs manquantes et des valeurs aberrantes")
            missing_data = data.isnull().sum()
            st.write(missing_data[missing_data > 0])

            st.write("Gestion des valeurs manquantes")
            data.dropna(inplace=True)
            st.write("Données après suppression des valeurs manquantes :")
            st.write(data.head())

            st.write("Sélection des features et de la cible")
            target_column = st.selectbox("Sélectionnez la colonne cible", data.columns)
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            st.warning("Veuillez uploader un fichier CSV pour préparer les données.")

    elif choice == "Modélisation":
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.header("Modélisation")
            target_column = st.selectbox("Sélectionnez la colonne cible", data.columns)
            X = data.drop(columns=[target_column])
            y = data[target_column]

            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns

            # Imputer les valeurs manquantes, puis appliquer le scaler
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

            # Imputer les valeurs manquantes pour les variables catégorielles
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            model_choice = st.selectbox("Choisissez le modèle", ["Régression Logistique", "Arbre de Décision"])

            if model_choice == "Régression Logistique":
                model = LogisticRegression(max_iter=1000)
            else:
                model = DecisionTreeClassifier()

            clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            st.header("Évaluation du modèle")
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)

            st.write(f"Accuracy: {accuracy}")
            st.write(f"F1 Score: {f1}")

            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Matrice de Confusion")
            st.plotly_chart(fig)
        else:
            st.warning("Veuillez uploader un fichier CSV pour modéliser les données.")

if __name__ == '__main__':
    main()
