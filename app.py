import streamlit as st

import dataset as dt
import algorithme as algo


@st.cache_data
def load_data():
    df = dt.X.copy()
    target_col = dt.y.columns[0]
    df[target_col] = dt.y.iloc[:, 0]
    return df, target_col


df, target_col = load_data()

# Config de la page
st.set_page_config(page_title="Analyse des Performances", layout="wide")

# CSS personnalisé pour le style
st.markdown("""
<style>
/* Fond noir et texte général */
body, .stApp {
    background-color: white;
    color: black;
}

/* Boutons stylés */
.stButton>button {
    background-color: orange;
    color: black;
    font-weight: bold;
    height: 50px;
    width: 150px;
    border-radius: 5px;
    font-size: 16px;
    margin: 5px;
}
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("<h2 style='text-align:center;'>Analyse des Performances des Étudiants</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Projet de Machine Learning sur les résultats universitaires</p>", unsafe_allow_html=True)

# État de la page
if "page" not in st.session_state:
    st.session_state.page = "APERÇU"

# Sélecteur horizontal
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("APERÇU"):
        st.session_state.page = "APERÇU"

with col2:
    if st.button("MODÈLE ML"):
        st.session_state.page = "MODÈLE ML"

with col3:
    if st.button("VISUALISATION"):
        st.session_state.page = "VISUALISATION"

with col4:
    if st.button("RAPPORTS"):
        st.session_state.page = "RAPPORTS"

# conteneur principal pour le contenu dynamique
main = st.container()

# Contenu “plein écran” selon la page active
with main:
    if st.session_state.page == "APERÇU":
        st.header("📊 Aperçu du Dataset")

        with st.container(border=True):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Nombre d'étudiants", df.shape[0])
            with col_b:
                st.metric("Nombre de variables explicatives", df.shape[1] - 1)
            with col_c:
                st.metric("Nombre de classes cibles", df[target_col].nunique())
        
        st.subheader("Description du jeu de données")
        st.markdown(
            """
        Le jeu de données **Higher Education Students Performance Evaluation** provient du
        dépôt UCI Machine Learning Repository ([Yilmaz & Şekeroğlu, 2019](https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation)).

        - Domaine : sciences sociales / éducation.
        - Objectif : **prédire la performance de fin de semestre** d'étudiants de l'enseignement supérieur
          à partir de leurs caractéristiques personnelles, familiales et académiques.
        - Instances : 145 étudiants.
        - Variables : 31 caractéristiques explicatives + 1 variable cible (note finale codée de 0 = Fail à 7 = AA).
        - Type de variables : majoritairement **catégorielles / ordinales** (âges, type de bourse, type de lycée,
          profession des parents, heures d'étude hebdomadaires, assiduité, etc.).
        - Valeurs manquantes : aucune.

        D'après la documentation UCI, les questions 1 à 10 portent sur les **caractéristiques personnelles**,
        les questions 11 à 16 sur la **situation familiale**, et les questions restantes sur les **habitudes
        d'étude et le comportement académique**.
        """
        )

        st.subheader("Aperçu des premières lignes")
        st.dataframe(df.head())

        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(include="all"))

        st.subheader("Répartition de la variable cible")
        st.bar_chart(df[target_col].value_counts())

    elif st.session_state.page == "MODÈLE ML":
        st.header("🤖 Modèle de Machine Learning")

        st.markdown(
            """
        Le modèle est un réseau de neurones **dense** implémenté avec **Keras** :

        - Entrée : 31 variables explicatives.
        - Couche cachée : `Dense(16, activation="relu")`.
        - Couche de sortie : `Dense(n_classes, activation="softmax")`.
        - Perte : `sparse_categorical_crossentropy`.
        - Optimiseur : `Adam (learning_rate=0.001)`.

        Vous pouvez lancer l'entraînement ci-dessous et visualiser les courbes.
        """
        )

        epochs = st.slider("Nombre d'époques d'entraînement", min_value=10, max_value=300, value=100, step=10)

        if st.button("Entraîner/réentraîner"):
            with st.spinner("Entraînement du modèle en cours..."):
                result = algo.train_model(epochs=epochs)

            st.success("Entraînement terminé.")
            st.metric("Exactitude (accuracy)", f"{result['accuracy'] * 100:.2f} %")
            st.metric("Perte (loss)", f"{result['loss']:.4f}")

            history = result["history"]
            if "loss" in history:
                st.subheader("Courbe de perte")
                st.line_chart(history["loss"])
            if "sparse_categorical_accuracy" in history:
                st.subheader("Courbe d'accuracy")
                st.line_chart(history["sparse_categorical_accuracy"])

    elif st.session_state.page == "VISUALISATION":
        st.header("📈 Visualisation interactive")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        st.subheader("1. Histogramme d'une variable")
        if numeric_cols:
            feature_hist = st.selectbox(
                "Choisissez une variable numérique", numeric_cols, key="hist_feature"
            )
            st.bar_chart(df[feature_hist].value_counts().sort_index())
        else:
            st.info("Aucune variable numérique disponible pour l'histogramme.")

        st.subheader("2. Nuage de points coloré par la cible")
        if len(numeric_cols) >= 2:
            col_x, col_y = st.columns(2)
            with col_x:
                x_feature = st.selectbox(
                    "Variable pour l'axe X", numeric_cols, index=0, key="scatter_x"
                )
            with col_y:
                y_feature = st.selectbox(
                    "Variable pour l'axe Y",
                    numeric_cols,
                    index=1 if len(numeric_cols) > 1 else 0,
                    key="scatter_y",
                )
            st.scatter_chart(df[[x_feature, y_feature]])
        else:
            st.info("Au moins deux variables numériques sont nécessaires pour le nuage de points.")

        st.subheader("3. Matrice de corrélation")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            st.dataframe(corr)
        else:
            st.info("Pas assez de variables numériques pour calculer une matrice de corrélation.")

    elif st.session_state.page == "RAPPORTS":
        st.header("📝 Rapports")
        st.markdown(
            """
        Cette section peut être utilisée pour :

        - Résumer les principaux résultats du modèle.
        - Décrire les facteurs qui semblent les plus corrélés avec la performance.
        - Proposer des recommandations pour les décideurs pédagogiques.

        Nous pouvons compléter ce texte avec vos propres conclusions
        après analyse des visualisations et des performances du modèle.
        """
        )