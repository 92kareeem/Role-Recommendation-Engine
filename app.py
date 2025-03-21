import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset with encoding handling
df = pd.read_csv('role_skills.csv', encoding='utf-8-sig')

# Debug: Print the column names and the first few rows of the DataFrame
print("Columns in the DataFrame:", df.columns.tolist())
print("First few rows of the DataFrame:")
print(df.head())

# Check if 'Skills' column exists
if 'Skills' not in df.columns:
    raise KeyError("The column 'Skills' does not exist in the DataFrame. Please check the CSV file.")

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
st.title('Data Science Role Recommendation Engine')

# Dropdown for role selection
role = st.selectbox('Select a Role', df['Role'])

if st.button('Recommend'):
    recommendations = get_recommendations(role)
    st.write('Top 3 Recommended Roles:')
    for rec in recommendations:
        st.write(rec)