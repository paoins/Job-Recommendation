# AI-Powered Job Recommendation System

A production-ready hybrid recommendation system that matches data science candidates with relevant job opportunities using machine learning.

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://job-recommendation-hdlofdwnwdxcsdsyvafmfj.streamlit.app/)
---

## Project Overview

This system analyzes **8,900+ real LinkedIn job postings** and provides personalized job recommendations using a sophisticated hybrid approach that combines:

1. **Content-Based Filtering** (60%): Matches candidates based on skills, experience level, and location preferences
2. **Collaborative Filtering** (40%): Uses matrix factorization (SVD) to discover latent patterns from 5,000+ simulated candidate-job interactions

### Key Results
- **Average Match Accuracy**: 73% (Precision@10)
- **Cold Start Coverage**: 100% (graceful fallback to content-based)
- **Recommendation Speed**: <2 seconds for 10,000+ jobs
- **Explainability**: Shows matching skills and skill gaps for every recommendation

---

## Live Demo

**[Try the Interactive App →]([https://job-recommendation-hdlofdwnwdxcsdsyvafmfj.streamlit.app/](https://job-recommendation-hdlofdwnwdxcsdsyvafmfj.streamlit.app/))**

Features:
- Custom candidate profile builder
- Hybrid AI recommendations
- Visual skill gap analysis
- Exportable results (CSV)

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/Data Science** | NumPy, pandas, scikit-learn, SciPy |
| **Algorithms** | TF-IDF, SVD Matrix Factorization, Jaccard Similarity |
| **Frontend** | Streamlit |
| **Data** | LinkedIn Jobs Dataset (Kaggle) |
| **Deployment** | Streamlit Cloud / Local |

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Input (Candidate Profile)         │
│              Skills | Experience | Location              │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼─────┐                   ┌────▼──────┐
    │ Content- │                   │Collaborative│
    │  Based   │                   │ Filtering  │
    │(60% wt.) │                   │ (40% wt.)  │
    └────┬─────┘                   └────┬───────┘
         │                              │
         │  ┌──────────────────────┐    │
         └─►│  Hybrid Combiner     │◄───┘
            │  (Weighted Average)  │
            └──────────┬───────────┘
                       │
              ┌────────▼─────────┐
              │  Top-K Rankings  │
              │  + Explanations  │
              └──────────────────┘
```

---

##  Project Structure

```
job-recommender/
├── notebooks/
│   └── job_recommender_system.ipynb    # Full development pipeline (Colab)
├── app.py                               # Streamlit application
├── utils/
│   ├── data_loader.py                  # Data loading & preprocessing
│   └── recommender.py                  # Recommendation algorithms
├── data/
│   └── processed/
│       ├── data_science_jobs.csv       # 8,900+ LinkedIn jobs
│       ├── candidates.csv              # 1,000 generated candidates
│       └── interactions.csv            # 5,000+ interactions
├── models/
│   └── cf_model.pkl                    # Trained SVD model
├── requirements.txt
└── README.md
```

---

##  Methodology

### 1. Data Collection & Preprocessing
- **Source**: LinkedIn Jobs Dataset (1.3M+ jobs)
- **Filtering**: Extracted 8,900 data science roles
- **Feature Engineering**: 
  - TF-IDF vectorization of job descriptions (500 features)
  - Multi-label binarization of skills (top 1,000 skills)
  - One-hot encoding of experience levels
  - Remote work flags

### 2. Candidate Simulation
Generated 1,000 realistic candidate profiles with:
- **Domain specialization**: ML, Data Engineering, BI, Backend
- **Experience-based skill distribution**: 3-30 skills based on seniority
- **Salary expectations**: Market-realistic ranges by level
- **Location preferences**: Top tech hubs + remote options

### 3. Interaction Generation
Simulated 5,000+ candidate applications where:
- Application probability ∝ skill match score
- Ratings (1-5 stars) correlate with match quality
- Ensures monotonic relationship for CF training

### 4. Content-Based Filtering
**Skill Matching:**
```python
score = 0.6 × coverage + 0.4 × jaccard

coverage = |candidate_skills ∩ job_skills| / |job_skills|
jaccard = |candidate_skills ∩ job_skills| / |candidate_skills ∪ job_skills|
```

**Experience Matching:**
- Perfect match: 1.0
- ±1 level: 0.7
- ±2 levels: 0.4
- ±3+ levels: 0.2

**Final Score:**
```
Overall = 0.5×skills + 0.3×experience + 0.2×location
```

### 5. Collaborative Filtering (SVD)
- **Algorithm**: Truncated SVD with 50 latent factors
- **Matrix**: 500 candidates × 2,000 jobs (sparse)
- **Explained Variance**: 68%
- **Benefits**: Discovers non-obvious patterns (e.g., "candidates who liked X also liked Y")

### 6. Hybrid Combination
```python
hybrid_score = 0.6×content_score + 0.4×cf_score
```

**Cold Start Strategy:**
- New users: 100% content-based
- Existing users: Hybrid approach
- Graceful degradation if CF fails

---

## Evaluation Results

### Metrics (on 20% test set)

| Model | Precision@10 | Recall@10 | Coverage |
|-------|--------------|-----------|----------|
| Content-Based | 0.68 | 0.42 | 100% |
| Hybrid | **0.73** | **0.47** | 100% |

### Key Findings
1. **Hybrid outperforms content-based by 7% in precision**
2. **No cold start failures** - content-based fallback works perfectly
3. **High coverage** - recommendations for all user types
4. **Fast inference** - <2s for full job corpus

---

## Installation & Usage

### Local Setup

```bash
# Clone repository
git clone https://github.com/paoins/job-recommender.git
cd job-recommender

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### Programmatic Usage

```python
from utils.data_loader import load_jobs_data, load_cf_model
from utils.recommender import get_job_recommendations

# Load data
jobs_df = load_jobs_data()
cf_model = load_cf_model()

# Define candidate profile
candidate = {
    'skills': ['python', 'machine learning', 'sql'],
    'experience_level': 'Mid Level',
    'preferred_locations': 'Remote, San Francisco',
    'open_to_remote': True
}

# Get recommendations
recommendations = get_job_recommendations(
    candidate,
    jobs_df,
    cf_model=cf_model,
    use_hybrid=True,
    top_k=10
)

print(recommendations[['job_title', 'company', 'hybrid_score']])
```

---

## Features

### For Candidates
- Custom profile builder with skill selection
- Experience level auto-detection
- Real-time recommendation generation
- Skill gap analysis (what skills to learn)
- Matching score breakdowns
- CSV export for tracking applications

### For Recruiters
- Candidate database with 1,000+ profiles
- Batch recommendation generation
- Filtering by location, experience, match score
- Exportable results for ATS systems

---

## Future Enhancements

### Short-term (1-2 months)
- [ ] Add salary prediction model
- [ ] Implement user authentication
- [ ] Track real user interactions
- [ ] Add company recommendation feature

### Long-term (3-6 months)
- [ ] Real-time job scraping pipeline
- [ ] A/B testing framework
- [ ] Multi-language support
- [ ] Mobile app (Flutter)
- [ ] Resume parser integration

---

## Key Learnings

### Technical Challenges Solved
1. **CSV Serialization Bug**: Lists converted to strings when saving to CSV
   - **Solution**: Used `ast.literal_eval()` for safe parsing
   
2. **Dimensionality Explosion**: 33,000+ unique skills
   - **Solution**: Reduced to top 1,000 skills (covers 95% of jobs)
   
3. **Cold Start Problem**: New users have no CF data
   - **Solution**: Hybrid approach with content-based fallback

4. **Sparse Matrix Memory**: 500×2,000 matrix too large
   - **Solution**: Used `scipy.sparse.csr_matrix` + Truncated SVD


---

## Dataset

**Source**: [LinkedIn Jobs & Skills 2024 (Kaggle)](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024)

**Statistics**:
- Total jobs in dataset: 1,348,454
- Data science jobs: 8,912
- Unique companies: 3,247
- Unique skills: 33,222 (reduced to 1,000)
- Date range: January 2024

**License**: Dataset used for educational/portfolio purposes

---

## Contributing

This is a portfolio project, but feedback is welcome! Feel free to:
- Open issues for bugs or suggestions
- Submit PRs for improvements
- Star the repo if you find it useful

---

## Author

**Achraf Baba**
- LinkedIn: [linkedin.com/in/achraf-baba](https://www.linkedin.com/in/achraf-baba-7b8726210/)
- Email: babachraf1@gmail.com

---

## Acknowledgments

- LinkedIn Jobs Dataset by Asaniczka (Kaggle)
- Streamlit team for the amazing framework
- scikit-learn contributors for ML tools
