# utils/recommender.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# CONTENT-BASED FUNCTIONS

def calculate_skill_match_score(candidate_skills, job_skills):
    """Calculate skill match between candidate and job using Jaccard + Coverage"""
    candidate_set = set(candidate_skills)
    job_set = set(job_skills)
    
    if not job_set:
        return 0.0
    
    intersection = len(candidate_set & job_set)
    union = len(candidate_set | job_set)
    
    # Hybrid: 60% coverage (how many job requirements met) + 40% Jaccard (overall similarity)
    jaccard_score = intersection / union if union > 0 else 0
    coverage_score = intersection / len(job_set) if len(job_set) > 0 else 0
    
    return (0.6 * coverage_score) + (0.4 * jaccard_score)

def calculate_experience_match(candidate_exp, job_exp):
    """Calculate experience level compatibility"""
    hierarchy = {
        'Internship': 0, 
        'Entry Level': 1, 
        'Mid Level': 2, 
        'Senior': 3, 
        'Leadership': 4,
        'Manager': 4  
    }
    
    cand_level = hierarchy.get(candidate_exp, 2)
    job_level = hierarchy.get(job_exp, 2)
    diff = abs(cand_level - job_level)
    
    # Perfect match=1.0, 1 level off=0.7, 2 levels=0.4, 3+=0.2
    mapping = {0: 1.0, 1: 0.7, 2: 0.4}
    return mapping.get(diff, 0.2)

def calculate_location_match(candidate_locations, job_location, candidate_remote):
    """Calculate location compatibility"""
    if pd.isna(job_location):
        return 0.5
    
    job_loc = str(job_location).lower()
    
    # Perfect match for remote jobs
    if 'remote' in job_loc and candidate_remote:
        return 1.0
    
    # Check if any preferred location matches
    candidate_locs = [loc.strip().lower() for loc in str(candidate_locations).split(',')]
    for loc in candidate_locs:
        if loc and (loc in job_loc or job_loc in loc):
            return 1.0
    
    # Partial credit if open to remote
    return 0.5 if candidate_remote else 0.3

def get_content_based_recommendations(candidate_profile, jobs_df, top_k=10):
    """
    Pure content-based recommendations
    """
    results = []
    
    for _, job in jobs_df.iterrows():
        # Calculate component scores
        skill_score = calculate_skill_match_score(
            candidate_profile['skills'], 
            job['skills_list']
        )
        
        exp_score = calculate_experience_match(
            candidate_profile['experience_level'], 
            job['experience_level']
        )
        
        loc_score = calculate_location_match(
            candidate_profile['preferred_locations'], 
            job['job_location'], 
            candidate_profile['open_to_remote']
        )
        
        # Weighted total (50% skills, 30% experience, 20% location)
        overall_score = (0.5 * skill_score) + (0.3 * exp_score) + (0.2 * loc_score)
        
        # Skill overlap details
        matching_skills = set(candidate_profile['skills']) & set(job['skills_list'])
        missing_skills = set(job['skills_list']) - set(candidate_profile['skills'])
        
        results.append({
            'job_id': job.get('job_id', ''),
            'job_title': job['job_title'],
            'company': job['company'],
            'location': job['job_location'],
            'experience_level': job['experience_level'],
            'overall_score': overall_score,
            'skill_score': skill_score,
            'experience_score': exp_score,
            'location_score': loc_score,
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'num_matching_skills': len(matching_skills),
            'num_missing_skills': len(missing_skills),
            'job_summary': job.get('job_summary', '')[:300] + "..." if pd.notna(job.get('job_summary', '')) else ""
        })
    
    recommendations_df = pd.DataFrame(results)
    recommendations_df = recommendations_df.sort_values('overall_score', ascending=False)
    
    return recommendations_df.head(top_k)

# COLLABORATIVE FILTERING FUNCTIONS

def get_cf_recommendations(candidate_id, cf_model, jobs_df, interactions_df, top_k=10):
    """
    Get collaborative filtering recommendations
    """
    if cf_model is None:
        return pd.DataFrame()
    
    # Get jobs already applied to
    applied_jobs = interactions_df[
        interactions_df['candidate_id'] == candidate_id
    ]['job_id'].tolist() if len(interactions_df) > 0 else []
    
    # Get recommendations from CF model
    try:
        cf_recs = cf_model.recommend_jobs(
            candidate_id,
            jobs_df,
            top_k=top_k,
            exclude_applied=applied_jobs
        )
        return cf_recs
    except Exception as e:
        print(f"CF recommendation error: {e}")
        return pd.DataFrame()

# HYBRID RECOMMENDER

def safe_minmax(series):
    """Normalize to 0-1 with division-by-zero protection"""
    if series.max() == series.min():
        return np.zeros(len(series))
    return (series - series.min()) / (series.max() - series.min())

def get_hybrid_recommendations(
    candidate_profile,
    jobs_df,
    cf_model=None,
    interactions_df=None,
    content_weight=0.6,
    cf_weight=0.4,
    top_k=10
):
    """
    Hybrid recommendation combining content-based + collaborative filtering
    """
    
    # STEP 1: Get content-based recommendations
    content_recs = get_content_based_recommendations(
        candidate_profile, 
        jobs_df, 
        top_k=100  
    )
    
    # STEP 2: Check if CF is available
    cf_available = (
        cf_model is not None and 
        interactions_df is not None and 
        len(interactions_df) > 0
    )
    
    # STEP 3: Handle CF unavailable (cold start / no model)
    if not cf_available:
        # Pure content-based
        content_recs = content_recs.head(top_k).copy()
        content_recs['content_score_norm'] = safe_minmax(content_recs['overall_score'])
        content_recs['cf_score_norm'] = 0.0
        content_recs['hybrid_score'] = content_recs['content_score_norm']
        content_recs['recommendation_type'] = 'Content-Based Only'
        
        return content_recs
    
    # STEP 4: Try to get CF recommendations
    candidate_id = candidate_profile.get('candidate_id', None)
    
    if candidate_id is None:
        # New user without ID ->  use content only
        content_recs = content_recs.head(top_k).copy()
        content_recs['content_score_norm'] = safe_minmax(content_recs['overall_score'])
        content_recs['cf_score_norm'] = 0.0
        content_recs['hybrid_score'] = content_recs['content_score_norm']
        content_recs['recommendation_type'] = 'Content-Based (New User)'
        
        return content_recs
    
    cf_recs = get_cf_recommendations(
        candidate_id,
        cf_model,
        jobs_df,
        interactions_df,
        top_k=100
    )
    
    # STEP 5: Merge and combine scores
    if len(cf_recs) == 0:
        # CF didn't work ->  fall back to content
        content_recs = content_recs.head(top_k).copy()
        content_recs['content_score_norm'] = safe_minmax(content_recs['overall_score'])
        content_recs['cf_score_norm'] = 0.0
        content_recs['hybrid_score'] = content_recs['content_score_norm']
        content_recs['recommendation_type'] = 'Content-Based (CF Failed)'
        
        return content_recs
    
    # STEP 6: Normalize both scores
    content_recs['content_score_norm'] = safe_minmax(content_recs['overall_score'])
    cf_recs['cf_score_norm'] = safe_minmax(cf_recs['cf_score'])
    
    # STEP 7: Merge on job_id
    merged = content_recs.merge(
        cf_recs[['job_id', 'cf_score_norm']],
        on='job_id',
        how='left'
    )
    
    # Fill missing CF scores with 0
    merged['cf_score_norm'] = merged['cf_score_norm'].fillna(0)
    
    # STEP 8: Calculate hybrid score
    merged['hybrid_score'] = (
        content_weight * merged['content_score_norm'] +
        cf_weight * merged['cf_score_norm']
    )
    
    merged['recommendation_type'] = 'Hybrid (Content + CF)'
    
    # STEP 9: Sort and return top K
    result = merged.sort_values('hybrid_score', ascending=False).head(top_k)
    
    return result

# MAIN RECOMMENDATION FUNCTION 

def get_job_recommendations(
    candidate_profile,
    jobs_df,
    cf_model=None,
    interactions_df=None,
    use_hybrid=True,
    top_k=10
):
    """
    Main recommendation function with automatic fallback logic
    """
    
    if use_hybrid and cf_model is not None:
        return get_hybrid_recommendations(
            candidate_profile,
            jobs_df,
            cf_model=cf_model,
            interactions_df=interactions_df,
            content_weight=0.6,
            cf_weight=0.4,
            top_k=top_k
        )
    else:
        # Pure content-based
        return get_content_based_recommendations(
            candidate_profile,
            jobs_df,
            top_k=top_k
        )