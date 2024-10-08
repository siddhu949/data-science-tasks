import pandas as pd

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
else:
    print("Data not loaded, unable to display.")
