# 📊 Social Marketing Analysis using NLP and Machine Learning

This project explores behavioral insights into contraceptive method usage and brand-switching patterns through text analysis, clustering, and machine learning. The notebook walks through natural language processing, EDA, vectorization, dimensionality reduction, customer segmentation, and classification modeling based on real customer feedback.

---

## 🧠 Project Objectives

- Understand key themes in user feedback about contraceptives.
- Cluster and visualize common perceptions and concerns.
- Predict customer behavior — loyal users vs. likely switchers.
- Provide insights for social marketing and policy refinement.

---

## 🚀 Project Workflow

### 1. **Data Upload & Text Preprocessing**
- Data was uploaded through Google Colab’s file interface.
- Preprocessing steps included text normalization, stopword removal, tokenization, and lowercasing to prepare textual responses for analysis.

### 2. **Exploratory Data Analysis**
- Visualizations were generated to show frequency of contraceptive types, switching patterns, and method expectations.
- TF-IDF and n-gram word clouds were created to highlight dominant themes in customer feedback.

### 3. **Vectorization & Clustering**
- Textual responses were vectorized using TF-IDF and embedded using **Sentence-BERT**.
- **UMAP** was used for dimensionality reduction, followed by **KMeans** clustering to group similar responses.
- Clusters were visualized interactively using **Plotly**.

### 4. **Dendrogram Analysis**
- Hierarchical clustering was applied to uncover nested relationships in contraceptive usage using dendrogram plots.

### 5. **Sankey Diagram & Radar Plot**
- A **Sankey diagram** was created to visualize switching flows between contraceptive types.
- Radar plots illustrated multidimensional comparisons between client types (e.g., loyal vs. switchers).

### 6. **Classification Models**
- Categorical labels were encoded, and models were trained using:
  - **Logistic Regression**
  - **Random Forest**
  - **XGBoost**
- Model performance was evaluated using accuracy, classification reports, and confusion matrices.
- Hyperparameters were optimized using **GridSearchCV**.

---

## 📦 Key Technologies & Tools

- **Natural Language Processing:** `nltk`, `re`, `TfidfVectorizer`, `CountVectorizer`
- **Embeddings & Clustering:** `SentenceTransformer`, `UMAP`, `KMeans`, `scipy`
- **Data Manipulation & Visualization:** `pandas`, `matplotlib`, `seaborn`, `plotly`, `WordCloud`
- **Machine Learning:** `scikit-learn`, `XGBoost`, `RandomForestClassifier`, `LogisticRegression`
- **Flow & Comparative Charts:** `pySankey`, `Radar Chart` with `matplotlib`
- **Model Persistence:** `joblib`
- **Environment:** Built in **Google Colab**

---

## 📈 Outputs & Insights

- Clustering revealed distinct groups of users based on their feedback.
- Sankey diagrams exposed transition paths among different contraceptive methods.
- Classification models predicted switching behavior with measurable accuracy.
- Visuals and statistics help inform interventions for social marketing.

---

## 💡 How to Use

1. Open the notebook in **Google Colab**.
2. Upload the dataset when prompted.
3. Execute cells step-by-step for data processing, visualization, and modeling.
4. Modify or expand the model as needed for new datasets or additional analysis.

---

## 👤 Author

**Wesley Monda Ong'eta**  
📍 Nairobi, Kenya  
📧 wesleymonda84@gmail.com  
📞 +254 794 641 433

---

> *For collaboration or contributions, feel free to fork this repository or reach out via email.*

