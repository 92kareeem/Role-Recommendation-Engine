import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.DataFrame({
    "Role": [
        "Data Scientist", "ML Engineer", "Data Analyst", "Data Engineer", 
        "AI Researcher", "Business Analyst", "NLP Engineer"
    ],
    "Skills": [
        "Python, Statistics, Machine Learning, Data Visualization",
        "Python, Machine Learning, Deployment, Algorithms",
        "SQL, Python, Data Visualization, Excel",
        "Python, SQL, ETL, Cloud Computing",
        "Python, Deep Learning, Machine Learning, Algorithms",
        "Excel, SQL, Data Visualization, Business Intelligence",
        "Python, NLP, Machine Learning, Deep Learning"
    ]
})

# Convert skills into a format suitable for vectorization
data["Skills"] = data["Skills"].apply(lambda x: x.replace(",", ""))
vectorizer = CountVectorizer()
skill_matrix = vectorizer.fit_transform(data["Skills"])

# Compute cosine similarity
similarity_matrix = cosine_similarity(skill_matrix)

def recommend_roles(selected_role):
    if selected_role not in data["Role"].values:
        return [], []
    
    role_index = data[data["Role"] == selected_role].index[0]
    similarity_scores = list(enumerate(similarity_matrix[role_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    recommended_roles = []
    common_skills_list = []
    selected_role_skills = set(data.iloc[role_index]["Skills"].split())
    
    for i in similarity_scores[1:4]:  # Top 3 roles
        role_name = data.iloc[i[0]]["Role"]
        role_skills = set(data.iloc[i[0]]["Skills"].split())
        common_skills = selected_role_skills.intersection(role_skills)
        
        recommended_roles.append(role_name)
        common_skills_list.append(", ".join(common_skills))
    
    return recommended_roles, common_skills_list

# Streamlit UI
st.set_page_config(page_title="🔍 AI-Powered Job Role Recommendation", layout="wide")
st.markdown(
    """
    <style>
        body {
            background-color: #f8f9fa;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 25px;
            padding: 12px 25px;
            border-radius: 12px;
        }
        .stSelectbox label {
            font-size: 25px;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 22px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚀 AI-Powered Job Role Recommendation Engine")
st.write("### Discover job roles based on skill similarity")

col1, col2 = st.columns([2, 3])
with col1:
    selected_role = st.selectbox("🔎 Select a job role:", data["Role"].values)
    if st.button("✨ Get Recommendations", use_container_width=True):
        recommendations, common_skills = recommend_roles(selected_role)
        if recommendations:
            with col2:
                st.success("### ✅ Top 3 Recommended Roles:")
                for idx, role in enumerate(recommendations):
                    st.write(f"- 🎯 {role}")
                    st.write(f"  🔹 Common Skills: {common_skills[idx]}")
        else:
            st.warning("⚠️ No recommendations found.")
st.sidebar.header("🔧 About the App")
st.sidebar.info("This application recommends the top 3 most similar roles based on required skills using Cosine Similarity.")

st.sidebar.markdown("📌 **How it works?**")
st.sidebar.write("- Select a role from the dropdown.\n- Click 'Recommend' to get the top 3 closest job roles.\n- Based on skill similarity, the best matches are displayed.")
# Footer with branding
st.markdown("---")
st.markdown("📌 Developed by Kareem for ClassMent | 🚀 Powered by AI")
