import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, auc, f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Titre du dashboard
st.set_page_config(page_title="Projet Data Science", layout="wide")

# Fonction pour charger les données
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

def apply_scaler(data, scaler_choice):
    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_choice == "RobustScaler":
        scaler = RobustScaler()
    elif scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))
    return pd.DataFrame(scaled_data, columns=data.select_dtypes(include=[np.number]).columns)

st.sidebar.image("logo.png", use_column_width=True)

# Uploader le fichier CSV dans la sidebar
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type="csv")

# Affichage de l'image du logo dans l'en-tête de la barre latérale

def main():
    menu = ["Accueil", "Compréhension des données", "Modélisation et évaluation"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Accueil":
        st.title("Projet Data Science - Prédiction du Risque de Défaut de Crédit")
        st.write("""
            ## Introduction
            Ce dashboard interactif permet de charger un fichier CSV, d'explorer les données,
            de les préparer, de construire des modèles de prédiction et d'évaluer leurs performances. 
            Dans ce projet, nous allons explorer et analyser un ensemble de données de 5960 observations avec 13 variables, afin de prédire la probabilité de défaut de crédit. Les données contiennent des informations sur les prêts, les hypothèques, les emplois, et d'autres variables financières et démographiques.        
        """)
        st.image("Franck.png", use_column_width=True)
    elif choice == "Compréhension des données":
        if uploaded_file is not None:
                data = load_data(uploaded_file)
                st.header("Aperçu des données")
                st.write(data.info())
                st.write(data.head())
                
                st.write("""
                    ## Description des Données

                    Les données contiennent les variables suivantes:

                    - **BAD** : Indicateur de défaut (1 = En défaut, 0 = Conforme)
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
                if 'BAD' in data.columns and data['BAD'].dtype in ['int64', 'float64']:
                    bins = [-1, 0, 1]  # Ajustez ces seuils selon vos besoins
                    labels = ['Conforme', 'En défaut']
                    data['BAD'] = pd.cut(data['BAD'], bins=bins, labels=labels, right=True)
                    data['BAD'] = data['BAD'].astype(str)
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
                    st.header("Analyse Exploratoire des Données")

                    st.subheader("Statistiques descriptives")
                    st.write(data.describe())

                    st.subheader("Distribution des variables (Univarié)")
                    selected_col = st.selectbox("Sélectionnez une colonne pour analyser sa distribution", data.columns)
                    if 'BAD' in selected_col:
                        fig_bad = px.histogram(data, x='BAD', title='Distribution de BAD')
                        st.plotly_chart(fig_bad)
                        comment_bad = "Nous remarquons qu'il y a beaucoup plus de personnes conforme que de personnes en defaut. Nous avons environs 5000  individus qui sont conforme et moins de 1500 qui sont en defaut ."
                        st.write(f"Commentaire : {comment_bad}")

                    if 'LOAN' in selected_col:
                        fig_bad = px.histogram(data, x='LOAN', title='Distribution de LOAN')
                        st.plotly_chart(fig_bad)
                        comment_bad = (
                            "La distribution de la variable LOAN est asymétrique à droite, "
                            "indiquant une concentration élevée de prêts dans les gammes inférieures avec un pic entre 10k et 20k. "
                            "Les valeurs s'étendent d'environ 5k à plus de 80k, mais les prêts au-delà de 60k sont très peu nombreux. "
                            "Cette distribution peut indiquer que la majorité des clients optent pour des prêts de petites à moyennes sommes, "
                            "ce qui pourrait être dû à une politique de prudence face aux risques associés à de grands montants prêtés."
                        )
                        st.write(f"Commentaire : {comment_bad}")

                    if 'MORTDUE' in selected_col:
                        fig_bad = px.histogram(data, x='MORTDUE', title="Distribution des Montant dû sur l'hypothèque existante")
                        st.plotly_chart(fig_bad)
                        comment_bad = (
                        "La distribution est fortement concentrée autour de valeurs basses avec un pic marqué près de 50k, "
                        "indiquant que la majorité des hypothèques dans cet ensemble ont des montants dus faibles. "
                        "Bien que la distribution s'étende à des valeurs plus élevées, dépassant 350k, ces cas sont nettement moins fréquents. "
                        "Cette caractéristique peut suggérer une moindre vulnérabilité globale au défaut sur ces hypothèques, "
                        "mais aussi pointer vers des risques significatifs liés aux rares montants élevés."
                    )
                        st.write(f"Commentaire : {comment_bad}")
                    if 'VALUE' in selected_col:
                        fig_bad = px.histogram(data, x='VALUE', title="Distribution de la Valeur de la propriété actuelle")
                        st.plotly_chart(fig_bad)
                        comment_bad = (
                        "La distribution montre une forte concentration des valeurs autour de moins de 200k, avec un pic marqué près de 100k, "
                        "suggérant que la majorité des propriétés ont une valeur relativement modeste. "
                        "La distribution s'étend jusqu'à des valeurs supérieures, atteignant 800k, mais avec une fréquence nettement décroissante, "
                        "indiquant que les propriétés de très haute valeur sont rares dans cet ensemble de données. "
                        "Cette répartition des valeurs peut influencer la capacité des emprunteurs à obtenir des prêts plus élevés et pourrait "
                        "être indicative de la stabilité financière globale des emprunteurs dans l'ensemble de données."
        )
                        st.write(f"Commentaire : {comment_bad}")
                    if 'REASON' in selected_col:
                        fig_bad = px.histogram(data, x='REASON', title="Distribution de la Raison de la demande de prêt")
                        st.plotly_chart(fig_bad)
                        comment_bad = "Nous remarquons qu'il y a une forte demande de pret pour des raisons de consolidation de dettes que pour une amélioration de l'habitat . Nous avons entre autre plus de 4000 individus contre 1500 ."
                        st.write(f"Commentaire : {comment_bad}")
                    if 'JOB' in selected_col:
                        fig_bad = px.histogram(data, x='JOB', title="Distribution du Type d'emploi")
                        st.plotly_chart(fig_bad)
                        comment_bad = (
                        "La catégorie 'Other' domine nettement la distribution, ce qui indique que la majorité des emprunteurs dans l'ensemble de données "
                        "ne rentrent pas dans les catégories d'emploi traditionnelles listées ou travaillent dans des secteurs variés. "
                        "Les professions 'office', 'Mgr'  et 'ProfExe' (professionnels exécutifs) sont également bien représentées, "
                        "suggérant une présence significative d'individus ayant probablement un niveau de revenu et de stabilité financière plus élevé. "
                        "Les catégories 'Self' (indépendants) et 'Sales' sont moins représentées, ce qui pourrait indiquer des niveaux de revenus inférieurs ou "
                        "une stabilité d'emploi moindre comparativement aux autres groupes."
                    )
                        st.write(f"Commentaire : {comment_bad}")
                    if 'YOJ' in selected_col:
                        fig_bad = px.histogram(data, x='YOJ', title='Distribution de YOJ')
                        st.plotly_chart(fig_bad)
                        comment_bad = (
        "La distribution des années sur le poste actuel montre un pic significatif pour les emprunteurs avec peu d'ancienneté, "
        "surtout entre 0 et 5 ans. Cela pourrait refléter une instabilité professionnelle pour une partie des emprunteurs, "
        "ce qui est un facteur à considérer dans l'évaluation du risque de crédit."
    )
                        st.write(f"Commentaire : {comment_bad}")

                    if 'DEROG' in selected_col:
                        fig_bad = px.histogram(data, x='DEROG', title='Distribution de DEROG')
                        st.plotly_chart(fig_bad)
                        comment_bad = (
        "La distribution montre que la majorité des emprunteurs n'ont pas de dérogations sur leur dossier de crédit. "
        "Les cas avec des dérogations sont très rares, signalant des exceptions plutôt que la norme. "
        "Cela suggère un profil de risque généralement faible pour la majorité des emprunteurs."
    )
                        st.write(f"Commentaire : {comment_bad}")

                    if 'DELINQ' in selected_col:
                        fig_bad = px.histogram(data, x='DELINQ', title="Distribution DELINQ")
                        st.plotly_chart(fig_bad)
                        comment_bad = (
        "La distribution indique que la plupart des emprunteurs n'ont aucun retard de paiement, "
        "avec une présence minoritaire d'emprunteurs ayant des incidents. Cela peut être interprété comme un signe de bonne santé financière globale."
    )
                        st.write(f"Commentaire : {comment_bad}")
                    if 'CLAGE' in selected_col:
                        fig_bad = px.histogram(data, x='CLAGE', title="Distribution de la CLAGE")
                        st.plotly_chart(fig_bad)
                        comment_bad = (
        "Le graphique montre que de nombreux emprunteurs possèdent des lignes de crédit bien établies, "
        "avec un pic entre 100 et 200 mois. Cette ancienneté peut être favorable pour l'évaluation de leur crédibilité."
    )
                        st.write(f"Commentaire : {comment_bad}")
                    if 'NINQ' in selected_col:
                        fig_bad = px.histogram(data, x='NINQ', title="Distribution de la Raison de la demande de prêt")
                        st.plotly_chart(fig_bad)
                        comment_bad = (
        "Cette variable montre que la majorité des emprunteurs ont peu ou pas de nouvelles enquêtes de crédit, "
        "ce qui suggère une activité de crédit modérée et potentiellement moins de risque de surendettement."
    )
                        st.write(f"Commentaire : {comment_bad}")
                    if 'CLNO' in selected_col:
                        fig_bad = px.histogram(data, x='CLNO', title="Distribution du CLNO")
                        st.plotly_chart(fig_bad)
                        comment_bad = (
        "La plupart des emprunteurs gèrent un nombre modéré de lignes de crédit, avec un pic notable entre 10 et 20. "
        "Cela indique une gestion de crédit relativement diversifiée sans aller vers une prolifération excessive."
    )
                        st.write(f"Commentaire : {comment_bad}")
                    if 'DEBTINC' in selected_col:
                        fig_bad = px.histogram(data, x='DEBTINC', title="Distribution du DEBTINC")
                        st.plotly_chart(fig_bad)
                        comment_bad = (
        "La distribution du ratio dette/revenu est extrêmement concentrée autour de faibles valeurs, "
        "montrant que la majorité des emprunteurs ont un faible endettement par rapport à leur revenu, "
        "ce qui est un indicateur positif pour la stabilité financière."
    )
                        st.write(f"Commentaire : {comment_bad}")


                    st.subheader("Relations entre les variables (Bivarié)")
                    var1 = st.selectbox("Sélectionnez la première colonne", data.columns)
                    var2 = st.selectbox("Sélectionnez la deuxième colonne", data.columns)
                if var1 and var2:
                    if data[var1].dtype in ['int64', 'float64'] and data[var2].dtype == 'object':
                        fig = px.box(data, x=var2, y=var1, title=f"Relation entre {var2} et {var1}")
                    elif data[var1].dtype == 'object' and data[var2].dtype in ['int64', 'float64']:
                        fig = px.box(data, x=var1, y=var2, title=f"Relation entre {var1} et {var2}")
                    elif data[var1].dtype == 'object' and data[var2].dtype == 'object':
                        fig = px.bar(data, x=var1, color=var2, title=f"Relation entre {var1} et {var2}")
                    else:
                        fig = px.scatter(data, x=var1, y=var2, title=f"Relation entre {var1} et {var2}", marginal_y="violin", marginal_x="box")
                    st.plotly_chart(fig)
                    comment_bi = st.text_area(f"Commentaire sur la relation entre {var1} et {var2}")
                    st.write(f"Votre commentaire : {comment_bi}")

        else:
               st.warning("Veuillez uploader un fichier CSV pour voir l'aperçu des données.")


    elif choice == "Modélisation et évaluation":
        st.header("Modélisation et évaluation des modèles")
        if uploaded_file is not None:
            data = load_data(uploaded_file)

            if 'BAD' in data.columns and data['BAD'].dtype in ['int64', 'float64']:
                # Transformation des données
                bins = [-1, 0, 1]  # Ajustez ces seuils selon vos besoins
                labels = ['Conforme', 'En défaut']
                data['BAD'] = pd.cut(data['BAD'], bins=bins, labels=labels, right=True)

                # Prétraitement des variables
                for col in ['REASON', 'JOB', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']:
                    if data[col].dtype in ['float64', 'int64']:
                        data[col].fillna(data[col].median(), inplace=True)
                    else:
                        data[col].fillna(data[col].mode()[0], inplace=True)

                columns_to_exclude = st.multiselect("Sélectionnez les variables à exclure", options=data.columns)
                data = data.drop(columns=columns_to_exclude)
                scaler_choice = st.selectbox("Choisissez le type de standardisation", ["StandardScaler", "RobustScaler", "MinMaxScaler"])
                data_scaled = apply_scaler(data, scaler_choice)

                # Sélection de la colonne cible et des données
                target_column = st.selectbox("Sélectionnez la colonne cible", data.columns)
                X = data.drop(columns=[target_column])
                y = data[target_column]

                # Configuration des pipelines pour les transformations numériques et catégoriques
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                categorical_features = X.select_dtypes(include=['object']).columns

                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())  # ou utilisez scaler_choice si vous voulez varier le scaler
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

                # Choix et configuration du modèle
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

                # Division des données et entrainement
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted')
                grid_search.fit(X_train, y_train)
                y_pred = grid_search.predict(X_test)

                # Évaluation du modèle
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', pos_label='En défaut')  # Spécifiez pos_label explicitement
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Matrice de Confusion")
                st.plotly_chart(fig)
                comment_bi = (
                    "Les variables DELINQ, DEROG, et DEBTINC sont essentielles pour le modèle car elles montrent directement la capacité "
                    "et le comportement de paiement de l'emprunteur, influençant fortement la prédiction du risque de défaut. "
                    "Les variables comme NINQ et CLNO, moins directement liées au risque de défaut, pourraient être omises pour simplifier le modèle. "
                    "YOJ et CLAGE offrent des contextes utiles sur la stabilité de l'emprunteur mais doivent être utilisées judicieusement."
                )
                st.write(f"Votre commentaire : {comment_bi}")

                if model_choice == "Régression Logistique":
                    model = LogisticRegression(max_iter=1000)
                    param_grid = {
                        'classifier__C': np.logspace(-4, 4, 20),  # étendue plus large pour C
                        'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                        'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none']
                    }
                else:
                    model = DecisionTreeClassifier()
                    param_grid = {
                        'classifier__max_depth': [5, 10, 20],
                        'classifier__min_samples_split': [2, 10, 20]
                    }

                clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

                # Division des données et entrainement du modèle
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted',verbose=1)
                grid_search.fit(X_train, y_train)
                y_pred = grid_search.predict(X_test)
                y_prob = grid_search.predict_proba(X_test)[:, 1]  # Assurez-vous que cela retourne les probabilités pour la classe positive

                # Calcul des métriques
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', pos_label='En défaut')

                # Tracé de la courbe ROC
                fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label='En défaut')
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
                fig_roc.update_layout(title="Courbe ROC",
                                    xaxis_title='Taux de Faux Positifs',
                                    yaxis_title='Taux de Vrais Positifs')

                # Tracé de la courbe de précision-rappel
                precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label='En défaut')
                fig_pr = go.Figure(data=go.Scatter(x=recall, y=precision, mode='lines'))
                fig_pr.update_layout(title="Courbe de Précision-Rappel",
                                    xaxis_title='Rappel',
                                    yaxis_title='Précision')

                # Affichage des résultats et des graphiques dans Streamlit
                st.write(f"Accuracy: {accuracy}")
                st.write(f"F1 Score: {f1}")
                st.write("Meilleurs hyperparamètres :", grid_search.best_params_)
                st.plotly_chart(fig_roc)
                st.plotly_chart(fig_pr)
            else:
                st.warning("La colonne 'BAD' est requise et doit être de type numérique.")
        else:
            st.warning("Veuillez uploader un fichier CSV pour modéliser les données.")
        


if __name__ == '__main__':
    main()

