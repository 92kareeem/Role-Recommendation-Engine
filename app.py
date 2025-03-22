import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset with encoding handling
df = pd.read_csv('role_skills.csv', encoding='utf-8-sig')

# Convert skills to a list of strings
df['Skills'] = df['Skills'].apply(lambda x: x.split(';'))

# Create a CountVectorizer to convert skills into a matrix of token counts
vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
skills_matrix = vectorizer.fit_transform(df['Skills'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(skills_matrix, skills_matrix)

# Function to get top 3 similar roles
def get_recommendations(role, cosine_sim=cosine_sim):
    idx = df[df['Role'] == role].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Exclude the role itself and get top 3
    role_indices = [i[0] for i in sim_scores]
    return df['Role'].iloc[role_indices]

# Streamlit UI
st.set_page_config(page_title='AI Job Recommender', layout='wide', page_icon='🧠')

st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>🔍 AI & Data Science Role Recommender</h1>
    <p style='text-align: center;'>Find the best matching job roles based on required skills.</p>
    <hr>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    role = st.selectbox('Select a Role:', df['Role'])
    if st.button('Recommend'): 
        recommendations = get_recommendations(role)
        
        st.markdown("""
        <h3 style='color: #2C3E50;'>✨ Top 3 Recommended Roles:</h3>
        """, unsafe_allow_html=True)
        
        for i, rec in enumerate(recommendations, 1):
            st.success(f"{i}. {rec}")
        
        st.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.header("🔧 About the App")
st.sidebar.info("This application recommends the top 3 most similar roles based on required skills using Cosine Similarity.")

st.sidebar.markdown("📌 **How it works?**")
st.sidebar.write("- Select a role from the dropdown.\n- Click 'Recommend' to get the top 3 closest job roles.\n- Based on skill similarity, the best matches are displayed.")
