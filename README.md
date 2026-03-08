# Tennis Player Similarity Analysis Using Python and SQL

This project analyzes professional ATP tennis match data and identifies similar player profiles
using a cosine similarity–based approach.

The goal of the project is to demonstrate data handling, feature engineering, similarity analysis,
SQL integration, visualization, and an interactive presentation layer, as required in the course project.

This project was developed as an individual final assignment for a Python & SQL course.

---

## Project Goal

- Transform match-level tennis data into player-level performance features  
- Analyze player performance by season and surface  
- Identify similar players using cosine similarity  
- Perform exploratory data analysis  
- Use SQL for data storage, aggregation, and querying  
- Visualize results using Python  
- Build an interactive dashboard using Streamlit  

---

## Dataset

**ATP Tennis 2000–2025**

- Source: Kaggle  
- Professional men’s tennis match data  
- Includes:
  - Players and opponents
  - Match results
  - Surface type and round
  - Rankings, points, and betting odds  

**Dataset download link:**  
https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull/versions/971?resource=download

---

## Technologies Used

- **Python** (Pandas, NumPy)
- **Scikit-learn** (StandardScaler, Cosine Similarity)
- **SQLite** (SQL queries and aggregations)
- **Matplotlib** (Visualizations)
- **Streamlit** (Interactive dashboard)

---

## Project Structure

```
tennis_similarity_project/
├── app/
│   └── app.py
├── notebooks/
│   ├── data_load_and_clean.ipynb
│   ├── feature_engineering_and_similarity.ipynb
│   └── sql_integration.ipynb
├── data/
│   └── raw/
│       └── atp_tennis.csv
├── outputs/
│   └── tables/
│       ├── matches_raw.csv
│       ├── player_matches.csv
│       └── player_features.csv
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order
```bash
data_load_and_clean.ipynb
feature_engineering_and_similarity.ipynb
sql_integration.ipynb
```

These notebooks clean the data, engineer features, compute player similarity,
and store results in a SQLite database.

---

### 3. Launch Streamlit dashboard
```bash
streamlit run app/app.py
```

---

## Key Findings

- Player similarity strongly depends on season and surface type  
- Cosine similarity effectively captures overall performance profiles  
- Single statistics are insufficient to describe player similarity  
- Feature-level comparison helps explain similarity scores  

---

## Author

- **Name:** Arda Uzer  
- **Course:** Python & SQL Final Project  