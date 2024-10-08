import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the movie dataset CSV file
csv_file_path = 'IMDb Movies India.csv'  # Make sure this is the correct path to your CSV file

# Loading the dataset into a pandas DataFrame with a different encoding
try:
    movie_data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading the data: {e}")

# Display the first few rows of the dataset
if 'movie_data' in locals():
    print(movie_data.head())
    
    # 1. Basic statistics of the dataset
    print("\nBasic Statistics:")
    print(movie_data.describe())

    # 2. Checking for missing values
    print("\nMissing Values:")
    print(movie_data.isnull().sum())

    # 3. Distribution of movie ratings
    plt.figure(figsize=(10, 6))
    sns.histplot(movie_data['Rating'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Number of Movies')
    plt.show()

    # 4. Analyzing the relationship between genres and ratings
    if 'Genre' in movie_data.columns:
        # Ensure that the Genre column has valid entries
        if movie_data['Genre'].notnull().all():
            movie_data['Genre'] = movie_data['Genre'].str.split(',')

            # Exploding the DataFrame to have one genre per row
            genre_ratings = movie_data.explode('Genre')
            
            # Dropping any rows with missing ratings after exploding
            genre_ratings = genre_ratings.dropna(subset=['Rating'])

            plt.figure(figsize=(12, 6))
            sns.boxplot(data=genre_ratings, x='Genre', y='Rating', palette='Set3')
            plt.title('Movie Ratings by Genre')
            plt.xticks(rotation=45)
            plt.xlabel('Genre')
            plt.ylabel('Rating')
            plt.show()
        else:
            print("The 'Genre' column contains null values.")
    else:
        print("Column 'Genre' not found in the dataset.")

    # 5. Analyzing top directors and their average movie ratings
    if 'Director' in movie_data.columns:
        top_directors = movie_data['Director'].value_counts().nlargest(10).index
        director_ratings = movie_data[movie_data['Director'].isin(top_directors)]
        average_ratings = director_ratings.groupby('Director')['Rating'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Rating', y='Director', data=average_ratings.sort_values(by='Rating', ascending=False), palette='viridis')
        plt.title('Average Movie Ratings by Top Directors')
        plt.xlabel('Average Rating')
        plt.ylabel('Director')
        plt.show()
    else:
        print("Column 'Director' not found in the dataset.")

    # 6. Analyzing the relationship between the number of votes and ratings
    if 'Votes' in movie_data.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=movie_data, x='Votes', y='Rating', alpha=0.5, color='orange')
        plt.title('Relationship Between Number of Votes and Movie Ratings')
        plt.xlabel('Number of Votes')
        plt.ylabel('Rating')
        plt.xscale('log')  # Log scale for better visibility
        plt.show()
    else:
        print("Column 'Votes' not found in the dataset.")

else:
    print("Data not loaded, unable to display.")
