from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'movie_recommendation_app'  # For session management

# Load dataset
df = pd.read_csv("movie_dataset.csv")

# Features for content-based filtering
features = ['keywords', 'cast', 'genres', 'director']

# Preprocessing
for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    return ' '.join(row[feature] for feature in features)

df['combined_features'] = df.apply(combine_features, axis=1)

# Vectorization
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(feature_matrix)

# Format currency function
def format_currency(value):
    if pd.isna(value) or value == 0:
        return "Not Available"
    return f"${int(value):,}"

# Format runtime function
def format_runtime(minutes):
    if pd.isna(minutes) or minutes == 0:
        return "Not Available"
    hours = int(minutes) // 60
    mins = int(minutes) % 60
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"

# Recommendation function
def recommend_movies(title, num_recommendations=6):
    if title not in df['title'].values:
        return []
    movie_index = df[df['title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in sorted_movies[1:num_recommendations+1]]
    return recommended_indices

# Function to get movie details by ID
def get_movie_details(movie_id):
    movie = df.iloc[movie_id]
    
    # Clean data and handle missing values
    details = {
        'title': movie['title'],
        'original_title': movie['original_title'] if movie['original_title'] != movie['title'] else '',
        'overview': movie['overview'] if not pd.isna(movie['overview']) else 'No overview available',
        'genres': movie['genres'],
        'release_date': movie['release_date'] if not pd.isna(movie['release_date']) else 'Unknown',
        'vote_average': round(float(movie['vote_average']), 1) if not pd.isna(movie['vote_average']) else 0,
        'vote_count': int(movie['vote_count']) if not pd.isna(movie['vote_count']) else 0,
        'budget': format_currency(movie['budget']),
        'revenue': format_currency(movie['revenue']),
        'runtime': format_runtime(movie['runtime']),
        'tagline': movie['tagline'] if not pd.isna(movie['tagline']) else '',
        'director': movie['director'] if not pd.isna(movie['director']) else 'Unknown',
        'cast': movie['cast'] if not pd.isna(movie['cast']) else '',
        'production_companies': movie['production_companies'] if not pd.isna(movie['production_companies']) else '',
        'status': movie['status'] if not pd.isna(movie['status']) else 'Unknown',
        'language': movie['original_language'] if not pd.isna(movie['original_language']) else 'Unknown',
        'popularity': round(float(movie['popularity']), 1) if not pd.isna(movie['popularity']) else 0,
        'keywords': movie['keywords'] if not pd.isna(movie['keywords']) else ''
    }
    
    # Calculate ROI if both budget and revenue are available
    if isinstance(movie['budget'], (int, float)) and isinstance(movie['revenue'], (int, float)) and movie['budget'] > 0:
        details['roi'] = ((movie['revenue'] - movie['budget']) / movie['budget']) * 100
    else:
        details['roi'] = None
        
    return details

# Routes
@app.route('/')
def home():
    # Get 10 popular movies for suggestions
    popular_movies = df.sort_values('popularity', ascending=False).head(10)['title'].tolist()
    return render_template('index.html', popular_movies=popular_movies)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie_title')
    # Store the search query in session
    session['last_search'] = movie_title
    
    if movie_title not in df['title'].values:
        return render_template('recommendations.html', movie_title=movie_title, recommendations=[], error=True)
    
    recommended_indices = recommend_movies(movie_title)
    recommended_movies = [
        {
            'id': idx,
            'title': df['title'].iloc[idx],
            'genres': df['genres'].iloc[idx],
            'vote_average': round(float(df['vote_average'].iloc[idx]), 1) if not pd.isna(df['vote_average'].iloc[idx]) else 0,
            'release_date': df['release_date'].iloc[idx].split('-')[0] if not pd.isna(df['release_date'].iloc[idx]) else 'Unknown',
        }
        for idx in recommended_indices
    ]
    return render_template('recommendations.html', movie_title=movie_title, recommendations=recommended_movies, error=False)

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    movie_details = get_movie_details(movie_id)
    # Get similar movies
    similar_movies = []
    
    if movie_id < len(df):
        similarity_scores = list(enumerate(cosine_sim[movie_id]))
        sorted_similar = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:5]  # Top 4 similar
        
        similar_movies = [
            {
                'id': idx, 
                'title': df['title'].iloc[idx],
                'similarity': round(score * 100)
            } 
            for idx, score in sorted_similar
        ]
    
    return render_template('movie_detail.html', movie=movie_details, similar_movies=similar_movies, last_search=session.get('last_search', ''))

# Error handling
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)