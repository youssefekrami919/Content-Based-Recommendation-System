# Content-Based-Recommendation-System

# Financial Literacy Content-Based Recommendation System

##  Overview
A sophisticated content-based recommendation system for financial literacy content that combines text analysis (TF-IDF) with categorical feature encoding to deliver personalized educational recommendations. The system handles both existing users and cold-start scenarios with an intuitive Streamlit web interface.

##  Key Features

### Core Recommendation Engine
- **Hybrid Feature Engineering**: Combines TF-IDF text analysis with one-hot encoding of categorical features
- **Content-Based Filtering**: Uses cosine similarity between user profiles and item features
- **Cold-Start Handling**: Creates user profiles from explicit preferences for new users
- **KNN Item-Based**: Additional item similarity recommendations using Nearest Neighbors
- **Unique Item Guarantee**: Ensures no duplicate recommendations

### User Interface
- **Streamlit Web App**: Modern, responsive interface with sidebar navigation
- **New User Registration**: Dynamic form for preference collection
- **Real-time Recommendations**: Instant personalized results
- **Downloadable Outputs**: Export recommendations as CSV

### Technical Features
- **Feature Caching**: Pre-trained models and feature matrices for fast inference
- **Modular Design**: Separate backend processing and frontend interface
- **Comprehensive Outputs**: Generates Top-10, Top-20, and KNN predictions

## 🏗️ Architecture

recommender-system/
├── code/
│ ├── content_based.py # Core recommendation algorithm
│ ├── main.py # Streamlit web interface
│ └── models/ # Cached feature models
├── data/
│ └── cleaned_financial_data.csv # Dataset
├── results/
│ └── tables/
│ └── content_based/ # Generated recommendations
└── requirements.txt # Dependencies
