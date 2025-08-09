#  K-Drama Recommendation System

A *lightweight content-based recommender* for K-Dramas, built using *pandas*, *scikit-learn*, and *numpy*.  
It suggests similar K-Dramas based on genres, tags, country, and cast metadata.

---

##  How It Works

1. *Data Cleaning* — Missing values filled, duplicates removed.  
2. *Feature Engineering* — Combines genres, tags, country, and cast into a single text field.  
3. *Vectorization* — TF-IDF transforms text into a weighted numeric representation.  
4. *Similarity Scoring* — Cosine similarity finds the most similar K-Dramas.  
5. *Top-N Retrieval* — Returns the top 5 most similar titles.  

---

##  Future Improvements

- Integrate *Korean NLP* for better tokenization.  
- Add *weighted features* (e.g., cast importance vs. genre).  
- Deploy a *Streamlit* or *FastAPI* interface.  
- Support *collaborative filtering* using user ratings.  

---


👨‍💻 *Author:* Divyanshi Rathore   
📧 Contact: dira160803@gmail.com
🌟 If you found this helpful, *star the repo*!
