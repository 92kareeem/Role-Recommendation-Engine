**Solution:**  
🔗 **Live Web App:** https://role-recommendation-engine.streamlit.app/

---

## **Approach & Methodology**
1. **Data Processing:**  
   - The dataset contains job roles and their respective required skills.
   - Skills are stored as a **semicolon-separated list** which is converted into a structured format.

2. **Vectorization of Skills:**  
   - We use **CountVectorizer** to transform the skills into a numerical format, enabling similarity computation.

3. **Computing Similarity:**  
   - We apply **Cosine Similarity** to measure the closeness between different job roles based on skill overlap.

4. **Recommendation System:**  
   - The system identifies the **top 3 most similar roles** to the selected role and displays them.

5. **Streamlit Web Interface:**  
   - The app provides an **interactive dropdown** for role selection.
   - Displays the **three closest roles** in a visually appealing format.

---

## **Why Cosine Similarity?**
✅ **Captures Skill Overlap:** It considers both the **presence and importance** of shared skills.
✅ **Handles High-Dimensional Data Well:** Unlike Jaccard Similarity, Cosine Similarity works efficiently with **sparse text data**.
✅ **Avoids Frequency Bias:** Unlike Euclidean distance, it measures the **direction of skill vectors** rather than raw counts.

---

🚀 **Outcome:** A **user-friendly and efficient** role recommendation engine tailored for Data Science and ML job seekers!

