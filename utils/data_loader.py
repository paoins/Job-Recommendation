# utils/data_loader.py

import pandas as pd
import streamlit as st
import pickle
import ast
from pathlib import Path

@st.cache_data
def load_jobs_data(file_path="data/processed/data_science_jobs.csv"):
    """Load and parse jobs data with proper skills parsing"""
    df = pd.read_csv(file_path)
    
    # Safely parse skills_list from string to list
    def safe_parse_skills(skills):
        if isinstance(skills, str):
            try:
                return ast.literal_eval(skills)
            except:
                return []
        elif isinstance(skills, list):
            return skills
        else:
            return []
    
    df['skills_list'] = df['skills_list'].apply(safe_parse_skills)
    return df

@st.cache_data
def load_candidates_data(file_path="data/processed/candidates.csv"):
    """Load and parse candidates data with proper skills parsing"""
    df = pd.read_csv(file_path)
    
    # Safely parse skills_list
    def safe_parse_skills(skills):
        if isinstance(skills, str):
            try:
                return ast.literal_eval(skills)
            except:
                return []
        elif isinstance(skills, list):
            return skills
        else:
            return []
    
    df['skills_list'] = df['skills_list'].apply(safe_parse_skills)
    return df

@st.cache_data
def load_interactions_data(file_path="data/processed/interactions.csv"):
    """Load interaction data for collaborative filtering"""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.warning("⚠️ Interactions data not found. Collaborative filtering unavailable.")
        return pd.DataFrame()

@st.cache_resource
def load_cf_model(model_path="models/cf_model.pkl"):
    """Load the trained collaborative filtering model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("⚠️ CF model not found. Using content-based only.")
        return None
    except Exception as e:
        st.error(f"Error loading CF model: {e}")
        return None

def get_all_unique_skills(jobs_df):
    """Extract all unique skills from jobs"""
    all_skills = set()
    for skills in jobs_df['skills_list']:
        if isinstance(skills, list):
            all_skills.update(skills)
    return sorted(list(all_skills))

@st.cache_data
def get_skill_frequencies(jobs_df):
    """Get skill frequency counts for recommendations"""
    all_skills_flat = []
    for skills in jobs_df['skills_list']:
        if isinstance(skills, list):
            all_skills_flat.extend(skills)
    return pd.Series(all_skills_flat).value_counts()