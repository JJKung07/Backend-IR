import pandas as pd
import sqlite3

# Connect to (or create) the SQLite database
conn = sqlite3.connect('resoures/db.db')

# Define your chunk size (number of rows per chunk)
chunksize = 50000  # You can adjust this value based on available memory

# Iterate through the CSV file in chunks
for chunk in pd.read_csv('resoures/recipes.csv', chunksize=chunksize):
    # Append each chunk to the SQLite table 'TableName'
    chunk.to_sql('recepies', conn, if_exists='append', index=False)

# Iterate through the CSV file in chunks
for chunk in pd.read_csv('resoures/reviews.csv', chunksize=chunksize):
    # Append each chunk to the SQLite table 'TableName'
    chunk.to_sql('reviews', conn, if_exists='append', index=False)

# Close the connection
conn.close()
