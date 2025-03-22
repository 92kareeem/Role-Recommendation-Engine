# Role-Recommendation-Engine
Deployed on Streamlit : https://role-recommendation-engine.streamlit.app/
#### Approach/Methodology:

used cosine similarity to measure the similarity between roles based on their skills. Cosine similarity is a metric used to determine how similar two entities are, irrespective of their size. It measures the cosine of the angle between two vectors projected in a multi-dimensional space.

#### Why Cosine Similarity?:

#### Steps
1. Cosine similarity is chosen because it is effective in text-based similarity tasks and is not affected by the magnitude of the vectors, making it suitable for comparing roles based on skill sets.

2. Convert the Dataset to CSV

3. Preprocessing: preprocess the data to calculate the similarity between roles based on shared skills.

4. Similarity Calculation: use cosine similarity to calculate the similarity between roles.

5. Streamlit UI: create an interactive Streamlit UI where users can input a role and get the top 3 recommended similar roles.
