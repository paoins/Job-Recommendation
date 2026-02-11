from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class CollaborativeRecommender:
    """
    Matrix factorization based collaborative filtering
    """

    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.user_to_idx = None
        self.job_to_idx = None
        self.idx_to_job = None

    def fit(self, interactions_df):
        print(" Training collaborative filtering model...")
        self.user_to_idx = {user: idx for idx, user in enumerate(interactions_df['candidate_id'].unique())}
        self.job_to_idx = {job: idx for idx, job in enumerate(interactions_df['job_id'].unique())}
        self.idx_to_job = {idx: job for job, idx in self.job_to_idx.items()}

        n_users = len(self.user_to_idx)
        n_jobs = len(self.job_to_idx)

        print(f"  Matrix size: {n_users} candidates × {n_jobs} jobs")

        rows = interactions_df['candidate_id'].map(self.user_to_idx)
        cols = interactions_df['job_id'].map(self.job_to_idx)
        data = interactions_df['rating'].values

        user_item_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_jobs))

        self.user_factors = self.svd.fit_transform(user_item_matrix)
        self.item_factors = self.svd.components_.T

        print(" Model trained!")
        print(f"  Explained variance: {self.svd.explained_variance_ratio_.sum():.2%}")
        return self

    def predict(self, candidate_id, job_id):
        if candidate_id not in self.user_to_idx or job_id not in self.job_to_idx:
            return 0.0
        user_idx = self.user_to_idx[candidate_id]
        job_idx = self.job_to_idx[job_id]
        return np.dot(self.user_factors[user_idx], self.item_factors[job_idx])

    def recommend_jobs(self, candidate_id, jobs_df, top_k=10, exclude_applied=None):
        if candidate_id not in self.user_to_idx:
            return pd.DataFrame()
        user_idx = self.user_to_idx[candidate_id]
        scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        job_ids = [self.idx_to_job[i] for i in range(len(scores))]
        recs = pd.DataFrame({
            'job_id': job_ids,
            'cf_score': scores
        })
        recs = recs.merge(
            jobs_df[['job_id', 'job_title', 'company', 'job_location', 'experience_level']],
            on='job_id'
        )
        if exclude_applied is not None:
            recs = recs[~recs['job_id'].isin(exclude_applied)]
        return recs.sort_values('cf_score', ascending=False).head(top_k)
