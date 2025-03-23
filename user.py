import random
from flask import Blueprint, request, jsonify
import sqlite3
import csv
from io import StringIO
from auth import token_required

user_bp = Blueprint('user', __name__)

DB_PATH = 'Resources/db5.db'

def parse_r_vector(value):
    if not value:
        return []
    cleaned = value.replace('c(', '').replace(')', '').strip()
    if not cleaned:
        return []
    reader = csv.reader(StringIO(cleaned), quotechar='"', skipinitialspace=True)
    try:
        items = next(reader)
    except StopIteration:
        items = []
    return [item.strip().strip('"') for item in items if item.strip()]

@user_bp.route('/api/user/folders', methods=['GET'])
@token_required
def get_folders(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Fetch folders with average rating
    cursor.execute('''
        SELECT f.FolderId, f.Name, AVG(b.Rating) as avg_rating
        FROM folders f
        LEFT JOIN bookmarks b ON f.FolderId = b.FolderId
        WHERE f.user_id = ?
        GROUP BY f.FolderId, f.Name
    ''', (user_id,))
    folders = cursor.fetchall()
    conn.close()
    folders_list = [
        {'id': row[0], 'name': row[1], 'avg_rating': row[2] if row[2] is not None else 0}
        for row in folders
    ]
    # Sort folders by avg_rating in descending order
    folders_list.sort(key=lambda x: x['avg_rating'], reverse=True)
    return jsonify(folders_list)

@user_bp.route('/api/user/folders', methods=['POST'])
@token_required
def create_folder(user_id):
    data = request.get_json()
    folder_name = data.get('name')
    if not folder_name:
        return jsonify({'error': 'Folder name is required'}), 400
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO folders (user_id, Name) VALUES (?, ?)', (user_id, folder_name))
        conn.commit()
        folder_id = cursor.lastrowid
        return jsonify({'id': folder_id, 'name': folder_name}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Folder name already exists'}), 409
    finally:
        conn.close()

@user_bp.route('/api/user/folders/<int:folder_id>', methods=['DELETE'])
@token_required
def delete_folder(user_id, folder_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT user_id FROM folders WHERE FolderId = ?', (folder_id,))
    folder = cursor.fetchone()
    if not folder or folder[0] != user_id:
        conn.close()
        return jsonify({'error': 'Folder not found or access denied'}), 404
    cursor.execute('DELETE FROM folders WHERE FolderId = ?', (folder_id,))
    cursor.execute('DELETE FROM bookmarks WHERE FolderId = ?', (folder_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Folder deleted successfully'}), 200

@user_bp.route('/api/user/folders/<int:folder_id>', methods=['PUT'])
@token_required
def update_folder(user_id, folder_id):
    data = request.get_json()
    new_name = data.get('name')
    if not new_name:
        return jsonify({'error': 'New folder name is required'}), 400
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT user_id FROM folders WHERE FolderId = ?', (folder_id,))
    folder = cursor.fetchone()
    if not folder or folder[0] != user_id:
        conn.close()
        return jsonify({'error': 'Folder not found or access denied'}), 404
    cursor.execute('UPDATE folders SET Name = ? WHERE FolderId = ?', (new_name, folder_id))
    conn.commit()
    conn.close()
    return jsonify({'id': folder_id, 'name': new_name}), 200

@user_bp.route('/api/user/bookmarks', methods=['POST'])
@token_required
def create_bookmark(user_id):
    data = request.get_json()
    folder_id = data.get('folderId')
    recipe_id = data.get('recipeId')
    rating = data.get('rating')
    if not folder_id or not recipe_id:
        return jsonify({'error': 'folderId and recipeId are required'}), 400
    if rating and (rating < 1 or rating > 5):
        return jsonify({'error': 'Rating must be between 1 and 5'}), 400
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT user_id FROM folders WHERE FolderId = ?', (folder_id,))
    folder = cursor.fetchone()
    if not folder or folder[0] != user_id:
        conn.close()
        return jsonify({'error': 'Invalid folder'}), 400
    cursor.execute('SELECT RecipeId FROM recipes WHERE RecipeId = ?', (recipe_id,))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'error': 'Recipe not found'}), 404
    try:
        cursor.execute('''
            INSERT INTO bookmarks (user_id, FolderId, RecipeId, Rating)
            VALUES (?, ?, ?, ?)
        ''', (user_id, folder_id, recipe_id, rating))
        conn.commit()
        return jsonify({'message': 'Bookmark added successfully'}), 201
    except sqlite3.IntegrityError as e:
        if 'UNIQUE constraint' in str(e):
            return jsonify({'error': 'Recipe already bookmarked in this folder'}), 409
        else:
            return jsonify({'error': 'Database error'}), 500
    finally:
        conn.close()

@user_bp.route('/api/user/bookmarks', methods=['GET'])
@token_required
def get_bookmarks(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Fetch bookmarks with folder details and calculate average rating
    cursor.execute('''
        SELECT b.FolderId, f.Name, b.RecipeId, r.Name, r.Images, b.Rating,
               AVG(b.Rating) OVER (PARTITION BY b.FolderId) as avg_folder_rating
        FROM bookmarks b
        JOIN folders f ON b.FolderId = f.FolderId
        JOIN recipes r ON b.RecipeId = r.RecipeId
        WHERE b.user_id = ?
    ''', (user_id,))
    bookmarks = cursor.fetchall()
    conn.close()

    # Organize bookmarks into folders
    folders_dict = {}
    for bm in bookmarks:
        folder_id = bm[0]
        if folder_id not in folders_dict:
            folders_dict[folder_id] = {
                'folderId': folder_id,
                'folderName': bm[1],
                'bookmarks': [],
                'avg_rating': bm[6] if bm[6] is not None else 0  # Average rating for the folder
            }
        images_list = parse_r_vector(bm[4])
        image = images_list[0] if images_list and images_list[0] != 'character(0)' else None
        folders_dict[folder_id]['bookmarks'].append({
            'recipeId': bm[2],
            'recipeName': bm[3],
            'image': image,
            'rating': bm[5] if bm[5] is not None else 0  # Default to 0 if no rating
        })

    # Convert to list and sort bookmarks within each folder by rating
    folders_list = list(folders_dict.values())
    for folder in folders_list:
        folder['bookmarks'].sort(key=lambda x: x['rating'], reverse=True)

    # Sort folders by average rating (descending)
    folders_list.sort(key=lambda x: x['avg_rating'], reverse=True)

    return jsonify(folders_list)

@user_bp.route('/api/user/bookmarks/<int:folder_id>/<int:recipe_id>', methods=['PUT'])
@token_required
def update_bookmark(user_id, folder_id, recipe_id):
    data = request.get_json()
    rating = data.get('rating')
    if rating is None or not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
        return jsonify({'error': 'Rating must be a number between 1 and 5'}), 400

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Verify ownership and existence
    cursor.execute('''
        SELECT b.user_id 
        FROM bookmarks b 
        WHERE b.FolderId = ? AND b.RecipeId = ? AND b.user_id = ?
    ''', (folder_id, recipe_id, user_id))
    bookmark = cursor.fetchone()

    if not bookmark:
        conn.close()
        return jsonify({'error': 'Bookmark not found or access denied'}), 404

    # Update rating
    cursor.execute('''
        UPDATE bookmarks 
        SET Rating = ? 
        WHERE FolderId = ? AND RecipeId = ? AND user_id = ?
    ''', (rating, folder_id, recipe_id, user_id))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Bookmark rating updated successfully'}), 200

@user_bp.route('/api/user/bookmarks/<int:folder_id>/<int:recipe_id>', methods=['DELETE'])
@token_required
def delete_bookmark(user_id, folder_id, recipe_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Verify ownership and existence
    cursor.execute('''
        SELECT b.user_id 
        FROM bookmarks b 
        WHERE b.FolderId = ? AND b.RecipeId = ? AND b.user_id = ?
    ''', (folder_id, recipe_id, user_id))
    bookmark = cursor.fetchone()

    if not bookmark:
        conn.close()
        return jsonify({'error': 'Bookmark not found or access denied'}), 404

    # Delete bookmark
    cursor.execute('''
        DELETE FROM bookmarks 
        WHERE FolderId = ? AND RecipeId = ? AND user_id = ?
    ''', (folder_id, recipe_id, user_id))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Bookmark removed successfully'}), 200

def get_recommendations_for_folder(folder_id):
    """Placeholder function to generate a list of recommended recipe IDs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Get recipe IDs already in the folder
    cursor.execute('SELECT RecipeId FROM bookmarks WHERE FolderId = ?', (folder_id,))
    bookmarked_ids = [row[0] for row in cursor.fetchall()]
    if not bookmarked_ids:
        conn.close()
        return []  # Return empty list if folder is empty
    # Get all recipe IDs not in the folder
    placeholders = ','.join('?' * len(bookmarked_ids))
    cursor.execute(f'SELECT RecipeId FROM recipes WHERE RecipeId NOT IN ({placeholders})', bookmarked_ids)
    all_ids = [row[0] for row in cursor.fetchall()]
    # Select up to 5 random recipe IDs as placeholder recommendations
    recommendation_ids = random.sample(all_ids, min(6, len(all_ids))) if all_ids else []
    conn.close()
    return recommendation_ids

@user_bp.route('/api/user/folders/<int:folder_id>/suggestions', methods=['GET'])
@token_required
def get_suggestions(user_id, folder_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Set row factory to return dict-like rows
    cursor = conn.cursor()

    # Verify folder ownership
    cursor.execute('SELECT user_id FROM folders WHERE FolderId = ?', (folder_id,))
    folder = cursor.fetchone()
    if not folder or folder[0] != user_id:
        conn.close()
        return jsonify({'error': 'Folder not found or access denied'}), 404

    # Get recommendations
    recommendation_ids = get_recommendations_for_folder(folder_id)
    if not recommendation_ids:
        conn.close()
        return jsonify({'error': 'Cannot generate suggestions for an empty folder'}), 400

    # Fetch recipe details
    placeholders = ','.join('?' * len(recommendation_ids))
    cursor.execute(f'''
        SELECT RecipeId AS id, 
               Name AS name,
               Description AS description,
               Keywords AS tags,
               Images AS images
        FROM recipes
        WHERE RecipeId IN ({placeholders})
    ''', recommendation_ids)
    recipes = cursor.fetchall()
    conn.close()

    # Format the response
    recipes_list = []
    for recipe in recipes:
        recipe_dict = dict(recipe)  # Now works because recipe is a Row object
        for field in ['images', 'tags']:
            recipe_dict[field] = parse_r_vector(recipe_dict.get(field, ''))
        recipes_list.append(recipe_dict)
    return jsonify(recipes_list)

@user_bp.route('/api/user/recommendations', methods=['GET'])
@token_required
def get_personalized_recommendations(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    folder_name_filter = request.args.get('folder_name', None)  # Optional query param
    try:
        # Get all bookmarked recipe IDs across all folders
        cursor.execute('SELECT DISTINCT RecipeId FROM bookmarks WHERE user_id = ?', (user_id,))
        all_bookmarked_ids = [row['RecipeId'] for row in cursor.fetchall()]

        # Get user's folders
        folder_query = 'SELECT FolderId, Name FROM folders WHERE user_id = ?'
        if folder_name_filter:
            folder_query += ' AND Name = ?'
            cursor.execute(folder_query, (user_id, folder_name_filter))
        else:
            cursor.execute(folder_query, (user_id,))
        folders = cursor.fetchall()
        if not folders:
            return jsonify({'error': 'No folders found'}), 404

        # Summary recommendations (from all bookmarks)
        summary_ids = random.sample(all_bookmarked_ids, min(6, len(all_bookmarked_ids))) if all_bookmarked_ids else []

        # Specific or random category recommendations
        if folder_name_filter:
            folder = folders[0]  # Assuming folder_name_filter matches exactly one
        else:
            folder = random.choice(folders)
        folder_id, folder_name = folder['FolderId'], folder['Name']

        # Fetch recommendations for this folder using existing function
        category_ids = get_recommendations_for_folder(folder_id)
        category_ids = random.sample(category_ids, min(6, len(category_ids))) if category_ids else []

        # Random dishes (not in any folder)
        if all_bookmarked_ids:
            placeholders = ','.join(['?'] * len(all_bookmarked_ids))
            query = f'SELECT RecipeId FROM recipes WHERE RecipeId NOT IN ({placeholders}) ORDER BY RANDOM() LIMIT 6'
            cursor.execute(query, all_bookmarked_ids)
        else:
            query = 'SELECT RecipeId FROM recipes ORDER BY RANDOM() LIMIT 6'
            cursor.execute(query)
        random_ids = [row['RecipeId'] for row in cursor.fetchall()]

        # Fetch recipe details
        all_ids = list(set(summary_ids + category_ids + random_ids))
        recipes = []
        if all_ids:
            placeholders = ','.join(['?'] * len(all_ids))
            cursor.execute(f'''
                SELECT RecipeId as id, Name as title, Description as description,
                       Keywords as tags, Images as images 
                FROM recipes WHERE RecipeId IN ({placeholders})
            ''', all_ids)
            recipes = [dict(row) for row in cursor.fetchall()]

        # Organize results
        recipe_map = {r['id']: r for r in recipes}

        return jsonify({
            'summary': [format_recipe(recipe_map[id]) for id in summary_ids if id in recipe_map][:6],
            'randomCategory': {
                'folderName': folder_name,
                'recipes': [format_recipe(recipe_map[id]) for id in category_ids if id in recipe_map][:6]
            },
            'randomDishes': [format_recipe(recipe_map[id]) for id in random_ids if id in recipe_map][:6]
        })

    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    finally:
        conn.close()

def format_recipe(recipe):
    return {
        **recipe,
        'tags': parse_r_vector(recipe.get('tags', '')),
        'images': parse_r_vector(recipe.get('images', ''))
    }