import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve
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
st.sidebar.image("logo.png", use_column_width=True)

# Uploader le fichier CSV dans la sidebar
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type="csv")

# Affichage de l'image du logo dans l'en-tête de la barre latérale

def main():
    menu = ["Accueil", "Compréhension des données", "Préparation des données", "Modélisation et évaluation"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Accueil":
        st.title("Projet Data Science - Prédiction du Risque de Défaut de Crédit")
        st.write("""
            ## Introduction
            Ce dashboard interactif permet de charger un fichier CSV, d'explorer les données,
            de les préparer, de construire des modèles de prédiction et d'évaluer leurs performances. 
            Dans ce projet, nous allons explorer et analyser un ensemble de données de 5960 observations avec 13 variables, afin de prédire la probabilité de défaut de crédit. Les données contiennent des informations sur les prêts, les hypothèques, les emplois, et d'autres variables financières et démographiques.        
        """)

    elif choice == "Compréhension des données":
        if uploaded_file is not None:
                data = load_data(uploaded_file)
                st.header("Aperçu des données")
                st.write(data.info())
                st.write(data.head())
                st.write("""
                    ## Description des Données

                    Les données contiennent les variables suivantes:

                    - **BAD** : Indicateur de défaut (1 = défaut, 0 = non défaut)
                    - **LOAN** : Montant du prêt
                    - **MORTDUE** : Montant dû sur l'hypothèque existante
                    - **VALUE** : Valeur de la propriété actuelle
                    - **REASON** : Raison de la demande de prêt (HomeImp = amélioration de l'habitat, DebtCon = consolidation de dettes)
                    - **JOB** : Type d'emploi
                    - **YOJ** : Nombre d'années à l'emploi actuel
                    - **DEROG** : Nombre de rapports dérogatoires majeurs
                    - **DELINQ** : Nombre de délais de paiement de 30 jours ou plus
                    - **CLAGE** : Âge moyen des lignes de crédit en mois
                    - **NINQ** : Nombre de demandes de crédit au cours des 6 derniers mois
                    - **CLNO** : Nombre de lignes de crédit
                    - **DEBTINC** : Ratio dette/revenu
                 
        """)

                st.header("Préparation des données")
                st.write("Vérification des valeurs manquantes et des valeurs aberrantes")
                missing_data = data.isnull().sum()
                st.write(missing_data[missing_data > 0])
                st.write(""" On remarque que bon nombre de variables ont des **valeurs manquantes**.
                    
                    * On a Montant dû sur l'hypothèque existante ici MORTDUE contient **518** valeurs manquantes.
                    * On a Valeur de la propriété actuelle ici VALUE contient **112** valeurs manquantes.
                    * On a Raison de la demande de prêt  ici REASON contient **252** valeurs manquantes.
                    * On a Type d'emploi ici JOB contient **279** valeurs manquantes.
                    * On a Nombre d'années à l'emploi actuel ici YOJ contient **515** valeurs manquantes.
                    * On a Nombre de rapports dérogatoires majeurs ici DEROG contient **708** valeurs manquantes.
                    * On a Nombre de délais de paiement de 30 jours ou plus ici DELINQ contient **580** valeurs manquantes.
                    * On a Âge moyen des lignes de crédit en mois ici CLAGE contient **308** valeurs manquantes.
                    * On a Nombre de demandes de crédit au cours des 6 derniers mois ici NINQ contient **510** valeurs manquantes.
                    * On a Nombre de lignes de crédit ici CLNO contient **221** valeurs manquantes.
                    * On a Indicateur de défaut  ici BAD contient **0** valeurs manquantes.
                    * On a Montant du prêt ici LOAN contient **0** valeurs manquantes.     
                          """)
                st.write("Gestion des valeurs manquantes")
                data['REASON'].fillna(data['REASON'].mode()[0], inplace=True)
                data['JOB'].fillna(data['JOB'].mode()[0], inplace=True)
                data['MORTDUE'].fillna(data['MORTDUE'].median(), inplace=True)
                data['VALUE'].fillna(data['VALUE'].median(), inplace=True)
                data['YOJ'].fillna(data['YOJ'].median(), inplace=True)
                data['DEROG'].fillna(data['DEROG'].median(), inplace=True)
                data['DELINQ'].fillna(data['DELINQ'].median(), inplace=True)
                data['CLAGE'].fillna(data['CLAGE'].median(), inplace=True)
                data['NINQ'].fillna(data['NINQ'].median(), inplace=True)
                data['CLNO'].fillna(data['CLNO'].median(), inplace=True)
                data['DEBTINC'].fillna(data['DEBTINC'].median(), inplace=True)

                st.write("Données après remplacement des valeurs manquantes par le mode et la mediane :")
                st.write(data.head())

        else:
               st.warning("Veuillez uploader un fichier CSV pour voir l'aperçu des données.")


    elif choice == "Préparation des données":
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

    elif choice == "Modélisation et évaluation":
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.header("Modélisation")
            target_column = st.selectbox("Sélectionnez la colonne cible", data.columns)
            X = data.drop(columns=[target_column])
            y = data[target_column]

            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

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
                param_grid = {
                    'classifier__C': [0.1, 1.0, 10],
                    'classifier__solver': ['liblinear', 'saga']
                }
            else:
                model = DecisionTreeClassifier()
                param_grid = {
                    'classifier__max_depth': [5, 10, 20],
                    'classifier__min_samples_split': [2, 10, 20]
                }

            clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted')
            grid_search.fit(X_train, y_train)
            y_pred = grid_search.predict(X_test)

            st.header("Évaluation du modèle")
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)

            st.write(f"Accuracy: {accuracy}")
            st.write(f"F1 Score: {f1}")
            st.write(f"Meilleurs hyperparamètres : {grid_search.best_params_}")

            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Matrice de Confusion")
            st.plotly_chart(fig)

            # Tracer les courbes de précision-rappel pour tous les modèles évalués
            st.header("Courbes de précision-rappel")

            fig = go.Figure()
            for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
                clf.set_params(**params)
                clf.fit(X_train, y_train)
                y_prob = clf.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=str(params)))

            fig.update_layout(title="Courbes de précision-rappel", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig)
        else:
            st.warning("Veuillez uploader un fichier CSV pour modéliser les données.")
    

if __name__ == '__main__':
    main()
