# user.py
from flask import Blueprint, request, jsonify
import sqlite3
import csv
from io import StringIO
from auth import token_required

user_bp = Blueprint('user', __name__)


def parse_r_vector(value):
    """Parse R-style vectors into proper Python lists"""
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
    conn = sqlite3.connect('Resources/db.db')
    cursor = conn.cursor()
    cursor.execute('SELECT FolderId, Name FROM folders WHERE user_id = ?', (user_id,))
    folders = cursor.fetchall()
    conn.close()
    folders_list = [{'id': row[0], 'name': row[1]} for row in folders]
    return jsonify(folders_list)


@user_bp.route('/api/user/folders', methods=['POST'])
@token_required
def create_folder(user_id):
    data = request.get_json()
    folder_name = data.get('name')
    if not folder_name:
        return jsonify({'error': 'Folder name is required'}), 400
    conn = sqlite3.connect('Resources/db.db')
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
    conn = sqlite3.connect('Resources/db.db')
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
    conn = sqlite3.connect('Resources/db.db')
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
    conn = sqlite3.connect('Resources/db.db')
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
    conn = sqlite3.connect('Resources/db.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT b.FolderId, f.Name, b.RecipeId, r.Name, r.Images, b.Rating
        FROM bookmarks b
        JOIN folders f ON b.FolderId = f.FolderId
        JOIN recipes r ON b.RecipeId = r.RecipeId
        WHERE b.user_id = ?
    ''', (user_id,))
    bookmarks = cursor.fetchall()
    conn.close()
    folders = {}
    for bm in bookmarks:
        folder_id = bm[0]
        if folder_id not in folders:
            folders[folder_id] = {
                'folderId': folder_id,
                'folderName': bm[1],
                'bookmarks': []
            }
        images_list = parse_r_vector(bm[4])
        image = images_list[0] if images_list and images_list[0] != 'character(0)' else None
        folders[folder_id]['bookmarks'].append({
            'recipeId': bm[2],
            'recipeName': bm[3],
            'image': image,
            'rating': bm[5]
        })
    return jsonify(list(folders.values()))


@user_bp.route('/api/user/bookmarks/<int:folder_id>/<int:recipe_id>', methods=['PUT'])
@token_required
def update_bookmark(user_id, folder_id, recipe_id):
    data = request.get_json()
    rating = data.get('rating')
    if rating is None or not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
        return jsonify({'error': 'Rating must be a number between 1 and 5'}), 400

    conn = sqlite3.connect('Resources/db.db')
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
    conn = sqlite3.connect('Resources/db.db')
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