import pandas as pd

# Path to the Titanic dataset CSV file
csv_file_path = 'tested.csv'  # Make sure 'tested.csv' is in the same directory or provide the full path

# Load the dataset into a pandas DataFrame
try:
    titanic_data = pd.read_csv(csv_file_path)
    print("Data loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found. Please check the path.")

# Display the first few rows of the dataset
if 'titanic_data' in locals():  # Check if the data was loaded
    print(titanic_data.head())
else:
    print("Data not loaded, unable to display.")
