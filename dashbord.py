import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go

# Set the page config
st.set_page_config(page_title="Projet Data Science", layout="wide")

# CSS for background styling
background_css = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #effbf0;
}
.main {
    background: url(https://github.com/Deadkeyz/dashboard/blob/main/bg.jpg);
    background-size: cover;
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] {
    background-color: #effbf0;
}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    data = data.sample(frac=1).reset_index(drop=True)
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

# Display the logo in the sidebar
st.sidebar.image("logo.png", use_column_width=True)


# File uploader
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type="csv")

def main():
    menu = ["Accueil", "Compréhension des données", "Modélisation et évaluation", "Amélioration du Modèle", "Conclusion"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Accueil":
        st.title("Projet Data Science - Analyse de Risque de Défaut sur Prêts Hypothécaires")
        st.write("""
            ## Introduction
            Lorsqu’on a des données, il est très important de les analyser en faisant une étude descriptive de celles-ci pour dégager tous les aspects importants pour notre étude. De les préparer, de construire des modèles de prédiction et d'évaluer leurs performances.
            Les données utilisées pour notre étude proviennent d’une base de données téléchargée sur le site Kaggle.
            Le jeu de données se nomme hmeq.csv.
            Il contient les données de 5 960 individus observés selon 13 variables présentées ci-dessous :
        """)
        
        # Display map with cities
        df_map = pd.DataFrame({
            'city': ['Abidjan', 'Bouaké', 'Daloa', 'Korhogo', 'Yamoussoukro', 'San-Pédro', 'Man', 'Gagnoa'],
            'lat': [5.30966, 7.6899, 6.8774, 9.4591, 6.8276, 4.7500, 7.4125, 6.1319],
            'lon': [-4.01266, -5.0318, -6.4502, -5.6296, -5.2767, -6.6500, -7.5536, -5.9498],
            'color': ['red', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange']
        })
        
        fig = go.Figure()
        
        for i, row in df_map.iterrows():
            fig.add_trace(go.Scattergeo(
                locationmode='country names',
                lon=[row['lon']],
                lat=[row['lat']],
                text=row['city'],
                marker=dict(
                    size=10,
                    color=row['color'],
                    line=dict(width=2, color='black')
                ),
                name=row['city']
            ))
        
        fig.update_layout(
            title_text='Pays des concepteurs du projet',
            showlegend=False,
            geo=dict(
                scope='africa',
                projection_type='natural earth',
                showland=True,
                landcolor='#effbf0',
                showocean=True,
                oceancolor='lightblue',
                lakecolor='lightblue',
                showcountries=True,
                countrycolor='black',
                lonaxis=dict(range=[-8.6, -2.5]),
                lataxis=dict(range=[4.0, 10.7]),
                resolution=50
            ),
            autosize=False,
            width=800,
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        st.plotly_chart(fig)    

    elif choice == "Compréhension des données":
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.header("Aperçu des données")
            st.write(data)
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
            st.header("Vérification des valeurs manquantes, doublons et des valeurs aberrantes")

            st.subheader("Es ce que le jeux de donnee contient des valeur manquante ?")

            missing_data = data.isnull().sum()
            st.write(missing_data[missing_data > 0])
            st.write(""" 
                On remarque que bon nombre de variables ont des **valeurs manquantes**.
                
                * Montant dû sur l'hypothèque existante (MORTDUE) contient **518** valeurs manquantes.
                * Valeur de la propriété actuelle (VALUE) contient **112** valeurs manquantes.
                * Raison de la demande de prêt (REASON) contient **252** valeurs manquantes.
                * Type d'emploi (JOB) contient **279** valeurs manquantes.
                * Nombre d'années à l'emploi actuel (YOJ) contient **515** valeurs manquantes.
                * Nombre de rapports dérogatoires majeurs (DEROG) contient **708** valeurs manquantes.
                * Nombre de délais de paiement de 30 jours ou plus (DELINQ) contient **580** valeurs manquantes.
                * Âge moyen des lignes de crédit en mois (CLAGE) contient **308** valeurs manquantes.
                * Nombre de demandes de crédit au cours des 6 derniers mois (NINQ) contient **510** valeurs manquantes.
                * Nombre de lignes de crédit (CLNO) contient **221** valeurs manquantes.
                * Indicateur de défaut (BAD) contient **0** valeurs manquantes.
                * Montant du prêt (LOAN) contient **0** valeurs manquantes.     
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
            
                st.write("Données après remplacement des valeurs manquantes par le mode et la médiane :")
                st.write(data)
                
                st.header("Est-ce que le jeu de données contient des doublons ?")
                duplicate_count = data.duplicated().sum()
                st.write(f"Nombre de doublons : {duplicate_count}")
            
                if duplicate_count > 0:
                    remove_duplicates = st.checkbox("Supprimer les doublons")
                    if remove_duplicates:
                        data = data.drop_duplicates()
                        st.write("Doublons supprimés.")
                        st.write(f"Nouveau nombre de doublons : {data.duplicated().sum()}")
                        st.write(data)
            
                st.header("Est-ce que le jeu de données contient des valeurs aberrantes ?")
                st.write("Les valeurs aberrantes seront détectées en utilisant les limites de l'écart interquartile (IQR).")
            
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                outlier_counts = {}
                for col in numeric_columns:
                    try:
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                        outlier_counts[col] = len(outliers)
                    except Exception as e:
                        st.write(f"Error processing column {col}: {e}")
            
                st.write("Nombre de valeurs aberrantes détectées par colonne :")
                st.write(outlier_counts)
            
                st.header("Analyse Exploratoire des Données")
            
                colors = ['#80b784', '#668d68', '#4d734d', '#335a33']

                # Fonction pour détecter les valeurs aberrantes
                def detect_outliers(data, col):
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                    return outliers

                st.subheader("Distribution des variables (Univarié)")
                univar_comments = {
                    'BAD': "Nous remarquons qu'il y a beaucoup plus de personnes conformes que de personnes en défaut. Nous avons environ 5000 individus qui sont conformes et moins de 1500 qui sont en défaut.",
                    'LOAN': "La distribution de la variable LOAN est asymétrique à droite, indiquant une concentration élevée de prêts dans les gammes inférieures avec un pic entre 10k et 20k. Les valeurs s'étendent d'environ 5k à plus de 80k, mais les prêts au-delà de 60k sont très peu nombreux. Cette distribution peut indiquer que la majorité des clients optent pour des prêts de petites à moyennes sommes, ce qui pourrait être dû à une politique de prudence face aux risques associés à de grands montants prêtés.",
                    'MORTDUE': "La distribution est fortement concentrée autour de valeurs basses avec un pic marqué près de 50k, indiquant que la majorité des hypothèques dans cet ensemble ont des montants dus faibles. Bien que la distribution s'étende à des valeurs plus élevées, dépassant 350k, ces cas sont nettement moins fréquents. Cette caractéristique peut suggérer une moindre vulnérabilité globale au défaut sur ces hypothèques, mais aussi pointer vers des risques significatifs liés aux rares montants élevés.",
                    'VALUE': "La distribution montre une forte concentration des valeurs autour de moins de 200k, avec un pic marqué près de 100k, suggérant que la majorité des propriétés ont une valeur relativement modeste. La distribution s'étend jusqu'à des valeurs supérieures, atteignant 800k, mais avec une fréquence nettement décroissante, indiquant que les propriétés de très haute valeur sont rares dans cet ensemble de données. Cette répartition des valeurs peut influencer la capacité des emprunteurs à obtenir des prêts plus élevés et pourrait être indicative de la stabilité financière globale des emprunteurs dans l'ensemble de données.",
                    'REASON': "Nous remarquons qu'il y a une forte demande de prêt pour des raisons de consolidation de dettes que pour une amélioration de l'habitat. Nous avons entre autres plus de 4000 individus contre 1500.",
                    'JOB': "La catégorie 'Other' domine nettement la distribution, ce qui indique que la majorité des emprunteurs dans l'ensemble de données ne rentrent pas dans les catégories d'emploi traditionnelles listées ou travaillent dans des secteurs variés. Les professions 'office', 'Mgr' et 'ProfExe' (professionnels exécutifs) sont également bien représentées, suggérant une présence significative d'individus ayant probablement un niveau de revenu et de stabilité financière plus élevé. Les catégories 'Self' (indépendants) et 'Sales' sont moins représentées, ce qui pourrait indiquer des niveaux de revenus inférieurs ou une stabilité d'emploi moindre comparativement aux autres groupes.",
                    'YOJ': "La distribution des années sur le poste actuel montre un pic significatif pour les emprunteurs avec peu d'ancienneté, surtout entre 0 et 5 ans. Cela pourrait refléter une instabilité professionnelle pour une partie des emprunteurs, ce qui est un facteur à considérer dans l'évaluation du risque de crédit.",
                    'DEROG': "La distribution montre que la majorité des emprunteurs n'ont pas de dérogations sur leur dossier de crédit. Les cas avec des dérogations sont très rares, signalant des exceptions plutôt que la norme. Cela suggère un profil de risque généralement faible pour la majorité des emprunteurs.",
                    'DELINQ': "La distribution indique que la plupart des emprunteurs n'ont aucun retard de paiement, avec une présence minoritaire d'emprunteurs ayant des incidents. Cela peut être interprété comme un signe de bonne santé financière globale.",
                    'CLAGE': "Le graphique montre que de nombreux emprunteurs possèdent des lignes de crédit bien établies, avec un pic entre 100 et 200 mois. Cette ancienneté peut être favorable pour l'évaluation de leur crédibilité.",
                    'NINQ': "Cette variable montre que la majorité des emprunteurs ont peu ou pas de nouvelles enquêtes de crédit, ce qui suggère une activité de crédit modérée et potentiellement moins de risque de surendettement.",
                    'CLNO': "La plupart des emprunteurs gèrent un nombre modéré de lignes de crédit, avec un pic notable entre 10 et 20. Cela indique une gestion de crédit relativement diversifiée sans aller vers une prolifération excessive.",
                    'DEBTINC': "La distribution du ratio dette/revenu est extrêmement concentrée autour de faibles valeurs, montrant que la majorité des emprunteurs ont un faible endettement par rapport à leur revenu, ce qui est un indicateur positif pour la stabilité financière."
                }
                
                for col, comment in univar_comments.items():
                    if col in data.columns:
                        try:
                            outliers = detect_outliers(data, col)
                            fig = px.histogram(data, x=col, title=f"Distribution de {col}", color_discrete_sequence=colors)
                            fig.add_trace(go.Histogram(
                                x=outliers[col],
                                name='Outliers',
                                marker=dict(color='red')
                            ))
                            st.plotly_chart(fig)
                            st.write(f"Commentaire : {comment}")
                        except Exception as e:
                            st.write(f"Error processing column {col}: {e}")
                
                st.subheader("Relations entre les variables (Bivarié)")

                bivar_comments = {
                    'LOAN': "La relation entre BAD et LOAN montre que les prêts plus élevés sont associés à un risque plus élevé de défaut. On observe que les individus en défaut (En défaut) ont tendance à avoir des montants de prêt plus élevés comparativement aux individus conformes (Conforme).",
                    'MORTDUE': "La relation entre BAD et MORTDUE montre que les montants dus sur les hypothèques sont plus élevés pour les individus en défaut. Cela pourrait indiquer une difficulté à gérer les obligations hypothécaires pour les personnes ayant des prêts en défaut.",
                    'VALUE': "La relation entre BAD et VALUE montre que la valeur des propriétés est légèrement plus basse pour les individus en défaut. Cela peut refléter une corrélation entre la valeur des biens possédés et la capacité à rembourser les prêts.",
                    'REASON': "La relation entre BAD et REASON montre que la majorité des défauts de paiement sont liés à des prêts pour la consolidation de dettes (DebtCon), plutôt que pour l'amélioration de l'habitat (HomeImp).",
                    'JOB': "La relation entre BAD et JOB montre que certaines catégories d'emplois sont plus susceptibles d'être en défaut que d'autres. Par exemple, les personnes dans des emplois de bureau (office) ou de gestion (Mgr) semblent avoir moins de défauts comparativement à d'autres catégories.",
                    'YOJ': "La relation entre BAD et YOJ (Years on Job) montre que les individus en défaut ont tendance à avoir une ancienneté moindre à leur emploi actuel. Cela pourrait indiquer une instabilité professionnelle comme un facteur de risque pour le défaut de paiement.",
                    'DEROG': "La relation entre BAD et DEROG montre que les individus avec un plus grand nombre de rapports dérogatoires sont plus susceptibles d'être en défaut. Cela souligne l'importance de l'historique de crédit dans l'évaluation du risque de défaut.",
                    'DELINQ': "La relation entre BAD et DELINQ montre que les individus ayant des délais de paiement plus fréquents sont également plus susceptibles d'être en défaut. Les retards de paiement sont donc un indicateur important du risque de défaut.",
                    'CLAGE': "La relation entre BAD et CLAGE montre que les individus avec des lignes de crédit plus anciennes sont moins susceptibles d'être en défaut. Une ancienneté plus grande des lignes de crédit peut indiquer une gestion plus stable et responsable du crédit.",
                    'NINQ': "La relation entre BAD et NINQ montre que les individus avec un plus grand nombre de nouvelles demandes de crédit dans les six derniers mois sont plus susceptibles d'être en défaut. Cela pourrait indiquer un besoin urgent de crédit, augmentant ainsi le risque de défaut.",
                    'CLNO': "La relation entre BAD et CLNO montre que les individus avec un nombre plus élevé de lignes de crédit sont plus susceptibles d'être en défaut. Une prolifération de lignes de crédit peut indiquer une gestion financière risquée.",
                    'DEBTINC': "La relation entre BAD et DEBTINC montre que les individus avec un ratio dette/revenu élevé sont plus susceptibles d'être en défaut. Un ratio dette/revenu élevé indique une charge financière importante par rapport aux revenus, augmentant ainsi le risque de défaut."
                }
                
                def detect_outliers(df, column):
                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                    return outliers
                
                for col, comment in bivar_comments.items():
                    if col in data.columns:
                        try:
                            if data[col].dtype in ['int64', 'float64']:
                                outliers = detect_outliers(data, col)
                                fig = px.box(data, x='BAD', y=col, title=f"Relation entre BAD et {col}")
                                fig.update_traces(marker=dict(color='#80b784'))
                                fig.add_trace(go.Box(
                                    y=outliers[col],
                                    x=outliers['BAD'],
                                    name='Outliers',
                                    marker=dict(color='red')
                                ))
                            else:
                                fig = px.bar(data, x='BAD', color=col, title=f"Relation entre BAD et {col}", color_discrete_sequence=px.colors.qualitative.Set1)
                            st.plotly_chart(fig)
                            st.write(f"Commentaire : {comment}")
                        except Exception as e:
                            st.write(f"Error processing column {col}: {e}")
                fig = px.strip(data, x='JOB', y='LOAN')
                st.plotly_chart(fig)
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

                # Choix et configuration des modèles
                model_choice = st.selectbox("Choisissez le modèle", ["Régression Logistique", "Arbre de Décision", "SVM", "Random Forest"])
                if model_choice == "Régression Logistique":
                    model = LogisticRegression(max_iter=1000)
                    param_grid = {
                        'classifier__C': [0.1, 1.0, 10],
                        'classifier__solver': ['liblinear', 'saga']
                    }
                elif model_choice == "Arbre de Décision":
                    model = DecisionTreeClassifier()
                    param_grid = {
                        'classifier__max_depth': [5, 10, 20],
                        'classifier__min_samples_split': [2, 10, 20]
                    }
                elif model_choice == "SVM":
                    model = SVC(probability=True)
                    param_grid = {
                        'classifier__C': [0.1, 1.0, 10],
                        'classifier__kernel': ['linear', 'rbf']
                    }
                else:
                    model = RandomForestClassifier()
                    param_grid = {
                        'classifier__n_estimators': [50, 100, 200],
                        'classifier__max_depth': [5, 10, 20]
                    }

                clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

                # Division des données et entrainement
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted')
                grid_search.fit(X_train, y_train)
                y_pred = grid_search.predict(X_test)
                y_prob = grid_search.predict_proba(X_test)[:, 1]  # Assurez-vous que cela retourne les probabilités pour la classe positive

                # Évaluation du modèle
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', pos_label='En défaut')  # Spécifiez pos_label explicitement
                cm = confusion_matrix(y_test, y_pred)

                st.write(f"Accuracy: {accuracy}, F1 Score: {f1}")
                cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale=[(0, "#102429"), (1, "#107d59")], title="Matrice de Confusion", template="plotly_white")
                st.plotly_chart(cm_fig)

                # Calcul des métriques
                fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label='En défaut')
                roc_auc = auc(fpr, tpr)
                roc_fig = px.area(x=fpr, y=tpr, title=f'Courbe ROC (AUC = {roc_auc:.2f})', labels=dict(x='Taux de Faux Positifs', y='Taux de Vrais Positifs'), template="plotly_white")
                roc_fig.update_traces(fillcolor="#107d59")
                st.plotly_chart(roc_fig)

                precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label='En défaut')
                pr_fig = px.area(x=recall, y=precision, title='Courbe de Précision-Rappel', labels=dict(x='Rappel', y='Précision'), template="plotly_white")
                pr_fig.update_traces(fillcolor="#107d59")
                st.plotly_chart(pr_fig)

                st.write("Meilleurs hyperparamètres :", grid_search.best_params_)
            else:
                st.warning("La colonne 'BAD' est requise et doit être de type numérique.")
        else:
            st.warning("Veuillez uploader un fichier CSV pour modéliser les données.")
    elif choice == "Amélioration du Modèle":
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.header("Amélioration du Modèle de Référence")
                st.write("""Dans cette partie au niveau du dataframe nous avons retirer les valeur aberantes 
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
                st.write("Les valeurs aberrantes seront détectées en utilisant les limites de l'écart interquartile (IQR).")

                numeric_columns = data.select_dtypes(include=[np.number]).columns
                outlier_counts = {}
                for col in numeric_columns:
                    try:
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                        outlier_counts[col] = len(outliers)
                        data = data[~((data[col] < lower_bound) | (data[col] > upper_bound))]
                    except Exception as e:
                        st.write(f"Error processing column {col}: {e}")

                st.write("Nombre de valeurs aberrantes détectées par colonne :")
                st.write(outlier_counts)
                st.header("Analyse Exploratoire des Données")

                colors = ['#80b784', '#668d68', '#4d734d', '#335a33']

                # Fonction pour détecter les valeurs aberrantes

                
                def detect_outliers(data, col):
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                    return outliers

                st.subheader("Distribution des variables (Univarié)")
                univar_comments = {
                    'BAD': "Nous remarquons qu'il y a beaucoup plus de personnes conformes que de personnes en défaut. Nous avons environ 5000 individus qui sont conformes et moins de 1500 qui sont en défaut.",
                    'LOAN': "La distribution de la variable LOAN est asymétrique à droite, indiquant une concentration élevée de prêts dans les gammes inférieures avec un pic entre 10k et 20k. Les valeurs s'étendent d'environ 5k à plus de 80k, mais les prêts au-delà de 60k sont très peu nombreux. Cette distribution peut indiquer que la majorité des clients optent pour des prêts de petites à moyennes sommes, ce qui pourrait être dû à une politique de prudence face aux risques associés à de grands montants prêtés.",
                    'MORTDUE': "La distribution est fortement concentrée autour de valeurs basses avec un pic marqué près de 50k, indiquant que la majorité des hypothèques dans cet ensemble ont des montants dus faibles. Bien que la distribution s'étende à des valeurs plus élevées, dépassant 350k, ces cas sont nettement moins fréquents. Cette caractéristique peut suggérer une moindre vulnérabilité globale au défaut sur ces hypothèques, mais aussi pointer vers des risques significatifs liés aux rares montants élevés.",
                    'VALUE': "La distribution montre une forte concentration des valeurs autour de moins de 200k, avec un pic marqué près de 100k, suggérant que la majorité des propriétés ont une valeur relativement modeste. La distribution s'étend jusqu'à des valeurs supérieures, atteignant 800k, mais avec une fréquence nettement décroissante, indiquant que les propriétés de très haute valeur sont rares dans cet ensemble de données. Cette répartition des valeurs peut influencer la capacité des emprunteurs à obtenir des prêts plus élevés et pourrait être indicative de la stabilité financière globale des emprunteurs dans l'ensemble de données.",
                    'REASON': "Nous remarquons qu'il y a une forte demande de prêt pour des raisons de consolidation de dettes que pour une amélioration de l'habitat. Nous avons entre autres plus de 4000 individus contre 1500.",
                    'JOB': "La catégorie 'Other' domine nettement la distribution, ce qui indique que la majorité des emprunteurs dans l'ensemble de données ne rentrent pas dans les catégories d'emploi traditionnelles listées ou travaillent dans des secteurs variés. Les professions 'office', 'Mgr' et 'ProfExe' (professionnels exécutifs) sont également bien représentées, suggérant une présence significative d'individus ayant probablement un niveau de revenu et de stabilité financière plus élevé. Les catégories 'Self' (indépendants) et 'Sales' sont moins représentées, ce qui pourrait indiquer des niveaux de revenus inférieurs ou une stabilité d'emploi moindre comparativement aux autres groupes.",
                    'YOJ': "La distribution des années sur le poste actuel montre un pic significatif pour les emprunteurs avec peu d'ancienneté, surtout entre 0 et 5 ans. Cela pourrait refléter une instabilité professionnelle pour une partie des emprunteurs, ce qui est un facteur à considérer dans l'évaluation du risque de crédit.",
                    'DEROG': "La distribution montre que la majorité des emprunteurs n'ont pas de dérogations sur leur dossier de crédit. Les cas avec des dérogations sont très rares, signalant des exceptions plutôt que la norme. Cela suggère un profil de risque généralement faible pour la majorité des emprunteurs.",
                    'DELINQ': "La distribution indique que la plupart des emprunteurs n'ont aucun retard de paiement, avec une présence minoritaire d'emprunteurs ayant des incidents. Cela peut être interprété comme un signe de bonne santé financière globale.",
                    'CLAGE': "Le graphique montre que de nombreux emprunteurs possèdent des lignes de crédit bien établies, avec un pic entre 100 et 200 mois. Cette ancienneté peut être favorable pour l'évaluation de leur crédibilité.",
                    'NINQ': "Cette variable montre que la majorité des emprunteurs ont peu ou pas de nouvelles enquêtes de crédit, ce qui suggère une activité de crédit modérée et potentiellement moins de risque de surendettement.",
                    'CLNO': "La plupart des emprunteurs gèrent un nombre modéré de lignes de crédit, avec un pic notable entre 10 et 20. Cela indique une gestion de crédit relativement diversifiée sans aller vers une prolifération excessive.",
                    'DEBTINC': "La distribution du ratio dette/revenu est extrêmement concentrée autour de faibles valeurs, montrant que la majorité des emprunteurs ont un faible endettement par rapport à leur revenu, ce qui est un indicateur positif pour la stabilité financière."
                }
                
                for col, comment in univar_comments.items():
                    if col in data.columns:
                        try:
                            outliers = detect_outliers(data, col)
                            fig = px.histogram(data, x=col, title=f"Distribution de {col}", color_discrete_sequence=colors)
                            fig.add_trace(go.Histogram(
                                x=outliers[col],
                                name='Outliers',
                                marker=dict(color='red')
                            ))
                            st.plotly_chart(fig)
                            st.write(f"Commentaire : {comment}")
                        except Exception as e:
                            st.write(f"Error processing column {col}: {e}")
                
                st.subheader("Relations entre les variables (Bivarié)")

                bivar_comments = {
                    'LOAN': "La relation entre BAD et LOAN montre que les prêts plus élevés sont associés à un risque plus élevé de défaut. On observe que les individus en défaut (En défaut) ont tendance à avoir des montants de prêt plus élevés comparativement aux individus conformes (Conforme).",
                    'MORTDUE': "La relation entre BAD et MORTDUE montre que les montants dus sur les hypothèques sont plus élevés pour les individus en défaut. Cela pourrait indiquer une difficulté à gérer les obligations hypothécaires pour les personnes ayant des prêts en défaut.",
                    'VALUE': "La relation entre BAD et VALUE montre que la valeur des propriétés est légèrement plus basse pour les individus en défaut. Cela peut refléter une corrélation entre la valeur des biens possédés et la capacité à rembourser les prêts.",
                    'REASON': "La relation entre BAD et REASON montre que la majorité des défauts de paiement sont liés à des prêts pour la consolidation de dettes (DebtCon), plutôt que pour l'amélioration de l'habitat (HomeImp).",
                    'JOB': "La relation entre BAD et JOB montre que certaines catégories d'emplois sont plus susceptibles d'être en défaut que d'autres. Par exemple, les personnes dans des emplois de bureau (office) ou de gestion (Mgr) semblent avoir moins de défauts comparativement à d'autres catégories.",
                    'YOJ': "La relation entre BAD et YOJ (Years on Job) montre que les individus en défaut ont tendance à avoir une ancienneté moindre à leur emploi actuel. Cela pourrait indiquer une instabilité professionnelle comme un facteur de risque pour le défaut de paiement.",
                    'DEROG': "La relation entre BAD et DEROG montre que les individus avec un plus grand nombre de rapports dérogatoires sont plus susceptibles d'être en défaut. Cela souligne l'importance de l'historique de crédit dans l'évaluation du risque de défaut.",
                    'DELINQ': "La relation entre BAD et DELINQ montre que les individus ayant des délais de paiement plus fréquents sont également plus susceptibles d'être en défaut. Les retards de paiement sont donc un indicateur important du risque de défaut.",
                    'CLAGE': "La relation entre BAD et CLAGE montre que les individus avec des lignes de crédit plus anciennes sont moins susceptibles d'être en défaut. Une ancienneté plus grande des lignes de crédit peut indiquer une gestion plus stable et responsable du crédit.",
                    'NINQ': "La relation entre BAD et NINQ montre que les individus avec un plus grand nombre de nouvelles demandes de crédit dans les six derniers mois sont plus susceptibles d'être en défaut. Cela pourrait indiquer un besoin urgent de crédit, augmentant ainsi le risque de défaut.",
                    'CLNO': "La relation entre BAD et CLNO montre que les individus avec un nombre plus élevé de lignes de crédit sont plus susceptibles d'être en défaut. Une prolifération de lignes de crédit peut indiquer une gestion financière risquée.",
                    'DEBTINC': "La relation entre BAD et DEBTINC montre que les individus avec un ratio dette/revenu élevé sont plus susceptibles d'être en défaut. Un ratio dette/revenu élevé indique une charge financière importante par rapport aux revenus, augmentant ainsi le risque de défaut."
                }
                
                def detect_outliers(df, column):
                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                    return outliers
                
                for col, comment in bivar_comments.items():
                    if col in data.columns:
                        try:
                            if data[col].dtype in ['int64', 'float64']:
                                outliers = detect_outliers(data, col)
                                fig = px.box(data, x='BAD', y=col, title=f"Relation entre BAD et {col}")
                                fig.update_traces(marker=dict(color='#80b784'))
                                fig.add_trace(go.Box(
                                    y=outliers[col],
                                    x=outliers['BAD'],
                                    name='Outliers',
                                    marker=dict(color='red')
                                ))
                            else:
                                fig = px.bar(data, x='BAD', color=col, title=f"Relation entre BAD et {col}", color_discrete_sequence=px.colors.qualitative.Set1)
                            st.plotly_chart(fig)
                            st.write(f"Commentaire : {comment}")
                        except Exception as e:
                            st.write(f"Error processing column {col}: {e}")
                st.header("Impact des variables sur BAD")

                st.header("Modélisation et évaluation des modèles")

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

                # Choix et configuration des modèles
                model_choice = st.selectbox("Choisissez le modèle", ["Régression Logistique", "Arbre de Décision", "SVM", "Random Forest"])
                if model_choice == "Régression Logistique":
                    model = LogisticRegression(max_iter=1000)
                    param_grid = {
                        'classifier__C': [0.1, 1.0, 10],
                        'classifier__solver': ['liblinear', 'saga']
                    }
                elif model_choice == "Arbre de Décision":
                    model = DecisionTreeClassifier()
                    param_grid = {
                        'classifier__max_depth': [5, 10, 20],
                        'classifier__min_samples_split': [2, 10, 20]
                    }
                elif model_choice == "SVM":
                    model = SVC(probability=True)
                    param_grid = {
                        'classifier__C': [0.1, 1.0, 10],
                        'classifier__kernel': ['linear', 'rbf']
                    }
                else:
                    model = RandomForestClassifier()
                    param_grid = {
                        'classifier__n_estimators': [50, 100, 200],
                        'classifier__max_depth': [5, 10, 20]
                    }

                clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

                # Division des données et entrainement
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted')
                grid_search.fit(X_train, y_train)
                y_pred = grid_search.predict(X_test)
                y_prob = grid_search.predict_proba(X_test)[:, 1]  # Assurez-vous que cela retourne les probabilités pour la classe positive

                # Évaluation du modèle
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', pos_label='En défaut')  # Spécifiez pos_label explicitement
                cm = confusion_matrix(y_test, y_pred)

                st.write(f"Accuracy: {accuracy}, F1 Score: {f1}")
                cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale=[(0, "#102429"), (1, "#107d59")], title="Matrice de Confusion", template="plotly_white")
                st.plotly_chart(cm_fig)

                # Calcul des métriques
                fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label='En défaut')
                roc_auc = auc(fpr, tpr)
                roc_fig = px.area(x=fpr, y=tpr, title=f'Courbe ROC (AUC = {roc_auc:.2f})', labels=dict(x='Taux de Faux Positifs', y='Taux de Vrais Positifs'), template="plotly_white")
                roc_fig.update_traces(fillcolor="#107d59")
                st.plotly_chart(roc_fig)

                precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label='En défaut')
                pr_fig = px.area(x=recall, y=precision, title='Courbe de Précision-Rappel', labels=dict(x='Rappel', y='Précision'), template="plotly_white")
                pr_fig.update_traces(fillcolor="#107d59")
                st.plotly_chart(pr_fig)

                st.write("Meilleurs hyperparamètres :", grid_search.best_params_)
            else:
                st.warning("La colonne 'BAD' est requise et doit être de type numérique.")
        else:
            st.warning("Veuillez uploader un fichier CSV pour modéliser les données.")
    
    elif choice == "Conclusion":
        st.header("Conclusion et Comparaison des Modèles")
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            
            # Vérifiez que la colonne 'BAD' est présente et de type numérique
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

                models = {
                    "Régression Logistique": LogisticRegression(max_iter=1000),
                    "Arbre de Décision": DecisionTreeClassifier(),
                    "SVM": SVC(probability=True),
                    "Random Forest": RandomForestClassifier()
                }

                param_grids = {
                    "Régression Logistique": {
                        'classifier__C': [0.1, 1.0, 10],
                        'classifier__solver': ['liblinear', 'saga']
                    },
                    "Arbre de Décision": {
                        'classifier__max_depth': [5, 10, 20],
                        'classifier__min_samples_split': [2, 10, 20]
                    },
                    "SVM": {
                        'classifier__C': [0.1, 1.0, 10],
                        'classifier__kernel': ['linear', 'rbf']
                    },
                    "Random Forest": {
                        'classifier__n_estimators': [50, 100, 200],
                        'classifier__max_depth': [5, 10, 20]
                    }
                }

                results = []
                
                # Division des données
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                for model_name, model in models.items():
                    param_grid = param_grids[model_name]
                    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
                    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted')
                    grid_search.fit(X_train, y_train)
                    y_pred = grid_search.predict(X_test)
                    y_prob = grid_search.predict_proba(X_test)[:, 1]

                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted', pos_label='En défaut')
                    precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label='En défaut')

                    results.append({
                        "Modèle": model_name,
                        "Accuracy": accuracy,
                        "F1 Score": f1,
                        "Recall": recall,
                        "Precision": precision,
                        "Meilleurs hyperparamètres": grid_search.best_params_
                    })

                results_df = pd.DataFrame(results).drop(columns=['Recall', 'Precision', 'Meilleurs hyperparamètres'])
                st.write(results_df)

                # Plotting comparison metrics
                st.header("Comparaison des modèles")
                fig = px.bar(results_df, x="Modèle", y=["Accuracy", "F1 Score"], barmode='group', title="Comparaison des modèles")
                fig.update_traces(marker_color=["#107d59", "#102429"], selector=dict(type='bar'))
                st.plotly_chart(fig)

                # Displaying detailed metrics for each model
                for result in results:
                    st.write(f"### {result['Modèle']}")
                    st.write(f"Meilleurs hyperparamètres : {result['Meilleurs hyperparamètres']}")
                    pr_fig = px.area(x=result['Recall'], y=result['Precision'], title=f'Courbe de Précision-Rappel ({result["Modèle"]})', labels=dict(x='Rappel', y='Précision'))
                    pr_fig.update_traces(fillcolor="#107d59")
                    st.plotly_chart(pr_fig)
            else:
                st.warning("La colonne 'BAD' est requise et doit être de type numérique.")
        else:
            st.warning("Veuillez uploader un fichier CSV pour modéliser les données.")

if __name__ == '__main__':
    main()
st.sidebar.image("logo-UFHB-e1699536639348-1024x747.png", use_column_width=True)
