import pandas as pd
import sqlite3

# Configuration
DB_PATH = 'resoures/db.db'
RECIPES_CSV = 'resoures/recipes.csv'
REVIEWS_CSV = 'resoures/reviews.csv'
CHUNKSIZE = 50000


def main():
    # Initialize database connection
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")

    # Create tables
    cursor = conn.cursor()

    # Create recipes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recipes (
            RecipeId INTEGER PRIMARY KEY,
            Name TEXT,
            AuthorId INTEGER,
            AuthorName TEXT,
            CookTime TEXT,
            PrepTime TEXT,
            TotalTime TEXT,
            DatePublished TEXT,
            Description TEXT,
            Images TEXT,
            RecipeCategory TEXT,
            Keywords TEXT,
            RecipeIngredientQuantities TEXT,
            RecipeIngredientParts TEXT,
            AggregatedRating REAL,
            ReviewCount REAL,
            Calories REAL,
            FatContent REAL,
            SaturatedFatContent REAL,
            CholesterolContent REAL,
            SodiumContent REAL,
            CarbohydrateContent REAL,
            FiberContent REAL,
            SugarContent REAL,
            ProteinContent REAL,
            RecipeServings REAL,
            RecipeYield TEXT,
            RecipeInstructions TEXT
        )
    ''')

    # Create reviews table with foreign key
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            ReviewId INTEGER,
            RecipeId INTEGER,
            AuthorId INTEGER,
            AuthorName TEXT,
            Rating INTEGER,
            Review TEXT,
            DateSubmitted TEXT,
            DateModified TEXT,
            FOREIGN KEY (RecipeId) REFERENCES recipes(RecipeId)
        )
    ''')
    conn.commit()

    # First pass: Load all valid RecipeIds from recipes.csv
    valid_recipe_ids = set()

    # Import recipes and collect IDs
    for chunk in pd.read_csv(RECIPES_CSV, chunksize=CHUNKSIZE):
        # Store valid IDs
        valid_recipe_ids.update(chunk['RecipeId'].unique())

        # Insert into database
        chunk.to_sql('recipes', conn, if_exists='append', index=False)

    print(f"Found {len(valid_recipe_ids)} valid Recipe IDs")

    # Second pass: Import reviews with validation
    total_reviews = 0
    rejected_reviews = 0

    for chunk in pd.read_csv(REVIEWS_CSV, chunksize=CHUNKSIZE):
        # Filter invalid RecipeIds
        original_count = len(chunk)
        valid_chunk = chunk[chunk['RecipeId'].isin(valid_recipe_ids)]
        filtered_count = len(valid_chunk)

        # Update counters
        rejected_reviews += (original_count - filtered_count)
        total_reviews += original_count

        # Insert valid reviews
        if not valid_chunk.empty:
            valid_chunk.to_sql('reviews', conn, if_exists='append', index=False)

    print(f"Processed {total_reviews} reviews")
    print(f"Rejected {rejected_reviews} reviews with invalid RecipeIds")

    # Verification query
    cursor.execute('''
        SELECT COUNT(*) FROM reviews
        WHERE RecipeId NOT IN (SELECT RecipeId FROM recipes)
    ''')
    orphaned = cursor.fetchone()[0]
    print(f"Orphaned reviews in database: {orphaned}")

    conn.close()


if __name__ == "__main__":
    main()