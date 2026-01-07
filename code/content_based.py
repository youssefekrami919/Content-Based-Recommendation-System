
# ============================================================================
# Content-Based Recommendation System (Enhanced CSV Output)
# Financial Literacy Dataset
# ============================================================================

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack

# ============================================================================
# 1. PATH CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
CONTENT_BASED_DIR = os.path.join(TABLES_DIR, 'content_based')

os.makedirs(CONTENT_BASED_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, 'cleaned_financial_data.csv')

# ============================================================================
# 2. LOAD DATA
# ============================================================================
df = pd.read_csv(DATA_FILE)
print("\n📊 Dataset Loaded (First 5 rows):")
print(df.head())

# ============================================================================
# 3. FEATURE EXTRACTION
# ============================================================================
text_cols = ['title', 'description', 'summary']
df['text_combined'] = df[text_cols].fillna('').agg(' '.join, axis=1)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
text_features = tfidf_vectorizer.fit_transform(df['text_combined'])

cat_cols = ['primary_topic', 'subtopic', 'difficulty', 'content_type']
try:
    ohe = OneHotEncoder(sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(sparse=False)

categorical_features = ohe.fit_transform(df[cat_cols])

knowledge_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
df['financial_knowledge_num'] = df['financial_knowledge'].map(knowledge_map)
num_features = df[['financial_knowledge_num']].values

item_features = hstack([text_features, categorical_features, num_features]).tocsr()
item_ids = df['item_id'].values

# Prepare item info dict to attach to recommendations
item_info_cols = ['item_id', 'title', 'primary_topic', 'subtopic', 'difficulty', 'content_type']
item_info_dict = df[item_info_cols].drop_duplicates(subset=['item_id']).set_index('item_id').to_dict('index')

print("\n📐 Item feature matrix shape:", item_features.shape)

# ============================================================================
# 4. USER PROFILE CONSTRUCTION
# ============================================================================
user_profiles = {}
user_ids = df['user_id'].unique()

for uid in user_ids:
    user_data = df[df['user_id'] == uid]
    ratings = user_data['rating'].values.reshape(-1, 1)
    indices = [np.where(item_ids == iid)[0][0] for iid in user_data['item_id']]
    user_profile = (item_features[indices].T @ ratings).flatten() / ratings.sum()
    user_profiles[uid] = user_profile

# Cold-start fallback
avg_profile = item_features.mean(axis=0).A1

print("\n👤 Sample User Profile (first 10 values):")
print(list(user_profiles.values())[0][:10])

# ============================================================================  
# 5. CONTENT-BASED RECOMMENDATIONS (Enhanced CSV, Fixed Top-N)  
# ============================================================================  
top_n_results = {10: [], 20: []}

for uid in user_ids:
    user_vec = user_profiles.get(uid, avg_profile).reshape(1, -1)
    sim_scores = cosine_similarity(user_vec, item_features).flatten()

    # Exclude already-rated items
    rated_items = set(df[df['user_id'] == uid]['item_id'].values)
    
    # Prepare list of items sorted by score descending, skip rated items
    items_scores = [(iid, score) for iid, score in zip(item_ids, sim_scores) if iid not in rated_items]
    
    # Keep only unique items per user
    seen_items = set()
    unique_items = []
    for iid, score in sorted(items_scores, key=lambda x: x[1], reverse=True):
        if iid not in seen_items:
            seen_items.add(iid)
            unique_items.append((iid, score))
        if len(unique_items) >= 20:  # max needed for Top-20
            break

    # Assign Top-10 and Top-20
    for N in [10, 20]:
        for iid, score in unique_items[:N]:
            info = item_info_dict[iid]
            top_n_results[N].append({
                'user_id': uid,
                'item_id': iid,
                'score': score,
                'title': info['title'],
                'primary_topic': info['primary_topic'],
                'subtopic': info['subtopic'],
                'difficulty': info['difficulty'],
                'content_type': info['content_type']
            })

# Save enhanced Top-N CSVs
for N in [10, 20]:
    df_out = pd.DataFrame(top_n_results[N])
    path = os.path.join(CONTENT_BASED_DIR, f'top_{N}_recommendations.csv')
    df_out.to_csv(path, index=False)
    print(f"\n📌 Top-{N} Recommendations (Enhanced CSV) saved at:", path)


# Save enhanced Top-N CSVs
for N in [10, 20]:
    df_out = pd.DataFrame(top_n_results[N])
    path = os.path.join(CONTENT_BASED_DIR, f'top_{N}_recommendations.csv')
    df_out.to_csv(path, index=False)
    print(f"\n📌 Top-{N} Recommendations (Enhanced CSV) saved at:", path)

# ============================================================================
# 6. ITEM-BASED k-NN (unchanged)
# ============================================================================
knn = NearestNeighbors(metric='cosine', algorithm='brute')  
knn.fit(item_features)  

knn_results = {}  
for k in [10, 20]:  
    distances, neighbors = knn.kneighbors(item_features, n_neighbors=k + 1)  
    knn_results[k] = {  
        item_ids[i]: [(item_ids[n], 1 - d) for n, d in zip(neighbors[i][1:], distances[i][1:])]  
        for i in range(len(item_ids))  
    }  

# Predict ratings for larger sample
avg_rating = df['rating'].mean()  
predictions = []  

for uid in user_ids[:2500]:  # first 2500 users  
    user_data = df[df['user_id'] == uid]  
    for iid in item_ids[:250]:  # first 250 items
        if iid in user_data['item_id'].values:  
            continue  # skip already rated items  

        sim_items = knn_results[20][iid]  # using k=20 
        weighted_sum, sim_sum = 0, 0  

        for sim_iid, sim_score in sim_items:  
            row = user_data[user_data['item_id'] == sim_iid]  
            if not row.empty:  
                weighted_sum += row['rating'].values[0] * sim_score  
                sim_sum += sim_score  

        pred_rating = weighted_sum / sim_sum if sim_sum > 0 else avg_rating  
        predictions.append({'user_id': uid, 'item_id': iid, 'pred_rating': pred_rating})  

# Convert to DataFrame and save  
pred_df = pd.DataFrame(predictions)  
pred_path = os.path.join(CONTENT_BASED_DIR, 'knn_predictions.csv')  
pred_df.to_csv(pred_path, index=False)  

print("\n📌 KNN Predictions (First 5 rows):")  
print(pred_df.head())

# ============================================================================
# 7. TF-IDF SAMPLE (unchanged)
# ============================================================================
full_tfidf_df = pd.DataFrame(text_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
full_tfidf_df['item_id'] = df['item_id'].values
full_tfidf_path = os.path.join(CONTENT_BASED_DIR, 'tfidf_all_items.csv')
full_tfidf_df.to_csv(full_tfidf_path, index=False)
print("\n📌 Full TF-IDF for all items (First 5 rows):")
print(full_tfidf_df.head())

# ============================================================================
print("\n✅ DONE — All outputs saved and previewed successfully.")
print("📁 Location:", CONTENT_BASED_DIR)
