# Train cf
from utils.cf_model import CollaborativeRecommender
from utils.data_loader import load_interactions_data, load_jobs_data

interactions_df = load_interactions_data()
jobs_df = load_jobs_data()

n_factors = min(30, interactions_df['job_id'].nunique() - 1)
cf_recommender = CollaborativeRecommender(n_factors=n_factors)
cf_recommender.fit(interactions_df)

# app.py

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import (
    load_jobs_data, 
    load_candidates_data, 
    load_interactions_data,
    load_cf_model,
    get_all_unique_skills
)
from utils.recommender import get_job_recommendations

# Page configuration
st.set_page_config(
    page_title="AI Job Recommender System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .job-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .skill-badge {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .missing-skill {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .rec-type-badge {
        background-color: #4caf50;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def display_metric_cards(recommendations, candidate_profile, rec_type):
    """Display metric cards at the top"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(recommendations)}</div>
            <div class="metric-label">Jobs Found</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Use hybrid_score if available, otherwise overall_score
        score_col = 'hybrid_score' if 'hybrid_score' in recommendations.columns else 'overall_score'
        avg_match = recommendations[score_col].mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_match:.0f}%</div>
            <div class="metric-label">Avg Match</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        score_col = 'hybrid_score' if 'hybrid_score' in recommendations.columns else 'overall_score'
        great_matches = len(recommendations[recommendations[score_col] > 0.7])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{great_matches}</div>
            <div class="metric-label">Great Matches (>70%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_skills = len(candidate_profile['skills'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_skills}</div>
            <div class="metric-label">Your Skills</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show recommendation type
    st.markdown(f'<div class="rec-type-badge"> {rec_type}</div>', unsafe_allow_html=True)

def display_job_card(job, index):
    """Display a single job recommendation card"""
    st.markdown(f"""
    <div class="job-card">
        <h3 style="margin:0; color:#1976d2;">#{index}  {job['job_title']}</h3>
        <h4 style="margin:0.5rem 0; color:#666;">{job['company']} •  {job['location']}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Display scores
    cols = st.columns(5)
    
    # Show hybrid score if available
    if 'hybrid_score' in job and pd.notna(job['hybrid_score']):
        cols[0].metric(" Hybrid Match", f"{job['hybrid_score']*100:.0f}%")
        cols[1].metric(" Content", f"{job.get('content_score_norm', 0)*100:.0f}%")
        cols[2].metric(" Collaborative", f"{job.get('cf_score_norm', 0)*100:.0f}%")
        cols[3].metric(" Experience", f"{job['experience_score']*100:.0f}%")
        cols[4].metric(" Location", f"{job['location_score']*100:.0f}%")
    else:
        cols[0].metric(" Overall Match", f"{job['overall_score']*100:.0f}%")
        cols[1].metric(" Skill Match", f"{job['skill_score']*100:.0f}%")
        cols[2].metric(" Experience", f"{job['experience_score']*100:.0f}%")
        cols[3].metric(" Location", f"{job['location_score']*100:.0f}%")
        cols[4].metric(" Skills Met", f"{job['num_matching_skills']}/{job['num_matching_skills']+job['num_missing_skills']}")
    
    # Display matching skills
    if job['matching_skills']:
        st.markdown("Your Matching Skills:")
        skills_html = " ".join([
            f'<span class="skill-badge">{skill}</span>' 
            for skill in job['matching_skills'][:15]
        ])
        st.markdown(skills_html, unsafe_allow_html=True)
    
    # Display missing skills
    if job['missing_skills'] and len(job['missing_skills']) > 0:
        st.markdown("Skills to Develop:")
        missing_html = " ".join([
            f'<span class="missing-skill">{skill}</span>' 
            for skill in job['missing_skills'][:10]
        ])
        st.markdown(missing_html, unsafe_allow_html=True)
    
    # Job description in expander
    if job['job_summary']:
        with st.expander(" View Full Job Description"):
            st.write(job['job_summary'])
    
    st.markdown("---")

def main():
    # Header
    st.markdown('<h1 class="main-header"> AI Job Recommender System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find your perfect data science role using hybrid machine learning</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner(" Loading job data and AI models..."):
        jobs_df = load_jobs_data()
        candidates_df = load_candidates_data()
        interactions_df = load_interactions_data()
        cf_model = load_cf_model()
        all_skills = get_all_unique_skills(jobs_df)
    
    # Show model status
    if cf_model is not None:
        st.success(" Hybrid AI Model Loaded (Content-Based + Collaborative Filtering)")
    else:
        st.info(" Using Content-Based Recommendations Only")
    
    # Sidebar - User Profile
    st.sidebar.title(" Your Profile")
    st.sidebar.markdown("---")
    
    # Model selection
    use_hybrid = st.sidebar.checkbox(
        " Use Hybrid AI Model",
        value=(cf_model is not None),
        disabled=(cf_model is None),
        help="Combines content-based filtering with collaborative filtering for better recommendations"
    )
    
    # Mode selection
    mode = st.sidebar.radio(
        "Choose Input Mode:",
        [" Create Custom Profile", " Use Existing Candidate"],
        help="Create your own profile or select from sample candidates"
    )
    
    candidate_profile = None
    
    if mode == " Create Custom Profile":
        st.sidebar.subheader("Build Your Profile")
        
        # Skills selection
        preferred_defaults = {
            "python", 
            "machine learning", 
            "sql", 
            "data analysis"
        }
        
        skill_map = {skill.lower(): skill for skill in all_skills}
        safe_defaults = [
            skill_map[s]
            for s in preferred_defaults
            if s in skill_map
        ]
        
        if len(safe_defaults) < 3:
            safe_defaults = all_skills[:3]
        
        user_skills = st.sidebar.multiselect(
            " Select Your Skills:",
            options=all_skills,
            default=safe_defaults,
            help="Choose 5–15 skills you have"
        )
        
        # Experience
        experience_years = st.sidebar.slider(
            " Years of Experience:",
            min_value=0,
            max_value=20,
            value=2,
            help="Your total years of professional experience"
        )
        
        # Map years to level
        if experience_years == 0:
            exp_level = "Internship"
        elif experience_years <= 2:
            exp_level = "Entry Level"
        elif experience_years <= 5:
            exp_level = "Mid Level"
        elif experience_years <= 10:
            exp_level = "Senior"
        else:
            exp_level = "Leadership"
        
        st.sidebar.info(f" Detected Level: {exp_level}")
        
        # Location preferences
        locations = st.sidebar.multiselect(
            " Preferred Locations:",
            ["Remote", "New York, NY", "San Francisco, CA", "Seattle, WA", 
             "Austin, TX", "Boston, MA", "Los Angeles, CA", "Chicago, IL",
             "Denver, CO", "Atlanta, GA"],
            default=["Remote"],
            help="Select your preferred work locations"
        )
        
        open_to_remote = st.sidebar.checkbox(
            " Open to Remote Work",
            value=True,
            help="Check if you're willing to work remotely"
        )
        
        candidate_profile = {
            'skills': user_skills,
            'experience_level': exp_level,
            'preferred_locations': ", ".join(locations),
            'open_to_remote': open_to_remote,
            'candidate_id': None  # New user, no CF history
        }
        
    else:  # Use existing candidate
        candidate_id = st.sidebar.selectbox(
            "Select a Candidate:",
            options=candidates_df['candidate_id'].tolist(),
            help="Choose from our sample candidates"
        )
        
        selected_candidate = candidates_df[
            candidates_df['candidate_id'] == candidate_id
        ].iloc[0]
        
        # Display profile info
        st.sidebar.subheader(" Profile Details")
        st.sidebar.write(f"Experience: {selected_candidate['experience_level']}")
        st.sidebar.write(f"Years: {selected_candidate['experience_years']}")
        st.sidebar.write(f"Education: {selected_candidate['education']}")
        st.sidebar.write(f"Domain: {selected_candidate.get('domain', 'N/A')}")
        st.sidebar.write(f"Total Skills: {len(selected_candidate['skills_list'])}")
        
        with st.sidebar.expander(" View All Skills"):
            st.write(", ".join(selected_candidate['skills_list'][:20]))
        
        candidate_profile = {
            'skills': selected_candidate['skills_list'],
            'experience_level': selected_candidate['experience_level'],
            'preferred_locations': selected_candidate['preferred_locations'],
            'open_to_remote': selected_candidate['open_to_remote'],
            'candidate_id': candidate_id  # Include for CF
        }
    
    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Recommendation Filters")
    
    num_recommendations = st.sidebar.slider(
        "Number of Results:",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )
    
    min_match_score = st.sidebar.slider(
        "Minimum Match Score:",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Only show jobs above this match threshold"
    )
    
    filter_location = st.sidebar.text_input(
        "Filter by Location:",
        placeholder="e.g., New York",
        help="Optional: filter results by location keyword"
    )
    
    # Get Recommendations Button
    st.sidebar.markdown("---")
    get_recs = st.sidebar.button("  Get Recommendations", type="primary")
    
    # Main content area
    if get_recs and candidate_profile:
        if len(candidate_profile['skills']) == 0:
            st.error(" Please select at least one skill!")
            return
        
        with st.spinner(" Analyzing thousands of jobs to find your perfect matches..."):
            # Get recommendations using hybrid or content-based
            recommendations = get_job_recommendations(
                candidate_profile,
                jobs_df,
                cf_model=cf_model if use_hybrid else None,
                interactions_df=interactions_df if use_hybrid else None,
                use_hybrid=use_hybrid,
                top_k=num_recommendations * 3  # Get extra for filtering
            )
            
            # Determine score column for filtering
            score_col = 'hybrid_score' if 'hybrid_score' in recommendations.columns else 'overall_score'
            
            # Apply filters
            recommendations = recommendations[
                recommendations[score_col] >= min_match_score
            ]
            
            if filter_location:
                recommendations = recommendations[
                    recommendations['location'].str.contains(
                        filter_location, 
                        case=False, 
                        na=False
                    )
                ]
            
            recommendations = recommendations.head(num_recommendations)
        
        # Display results
        if len(recommendations) == 0:
            st.warning(" No jobs found matching your criteria. Try adjusting your filters!")
            return
        
        # Get recommendation type
        rec_type = recommendations['recommendation_type'].iloc[0] if 'recommendation_type' in recommendations.columns else 'Content-Based'
        
        # Metrics
        st.markdown("##  Overview")
        display_metric_cards(recommendations, candidate_profile, rec_type)
        
        st.markdown("---")
        
        # Job recommendations
        st.markdown("##  Your Personalized Recommendations")
        st.markdown(f"Showing {len(recommendations)} jobs ranked by relevance")
        
        for idx, (_, job) in enumerate(recommendations.iterrows(), 1):
            display_job_card(job, idx)
        
        # Download button
        st.markdown("---")
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label=" Download Recommendations as CSV",
            data=csv,
            file_name="job_recommendations.csv",
            mime="text/csv"
        )
    
    else:
        # Welcome screen
        st.info("Get Started: Configure your profile in the sidebar and click 'Get Recommendations'")
        
        # Dataset statistics
        st.markdown("##  Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Jobs",
                f"{len(jobs_df):,}",
                help="Number of data science jobs in our database"
            )
        
        with col2:
            st.metric(
                "Companies",
                f"{jobs_df['company'].nunique():,}",
                help="Unique companies hiring"
            )
        
        with col3:
            unique_skills = len(get_all_unique_skills(jobs_df))
            st.metric(
                "Unique Skills",
                f"{unique_skills:,}",
                help="Different skills across all jobs"
            )
        
        # Model info
        if cf_model is not None:
            st.markdown("###  AI Model Information")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                Collaborative Filtering Model
                - Trained on {len(interactions_df):,} candidate-job interactions
                - Uses matrix factorization (SVD)
                - Discovers latent preferences
                """)
            with col2:
                st.info(f"""
                Content-Based Model
                - Analyzes {len(jobs_df):,} job descriptions
                - Matches skills, experience, location
                - Explainable recommendations
                """)
        
        # Top skills chart
        st.markdown("###  Most In-Demand Skills")
        
        all_skills_flat = [skill for skills in jobs_df['skills_list'] for skill in skills]
        top_skills = pd.Series(all_skills_flat).value_counts().head(15)
        
        st.bar_chart(top_skills)
        
        # Sample jobs
        st.markdown("###  Sample Job Postings")
        sample_jobs = jobs_df[['job_title', 'company', 'job_location', 'experience_level']].head(10)
        st.dataframe(sample_jobs, use_container_width=True)

if __name__ == "__main__":
    main()