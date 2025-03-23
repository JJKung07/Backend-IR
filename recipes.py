# recipes.py
import csv
from io import StringIO
from flask import Blueprint, jsonify
import sqlite3
from pathlib import Path

recipes_bp = Blueprint('recipes', __name__, url_prefix='/api')

DB_PATH = 'Resources/db5.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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

@recipes_bp.route('/recipes', methods=['GET'])
def get_recipes():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT RecipeId AS id, 
                   Name AS title,
                   Description AS description,
                   Keywords AS tags,
                   Images AS images,
                   DatePublished
            FROM recipes
            LIMIT 6
        ''')
        recipes = cursor.fetchall()
        conn.close()

        recipes_list = []
        for recipe in recipes:
            recipe_dict = dict(recipe)
            for field in ['images', 'tags']:
                recipe_dict[field] = parse_r_vector(recipe_dict.get(field, ''))
            del recipe_dict['DatePublished']
            recipes_list.append(recipe_dict)
        return jsonify(recipes_list)
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@recipes_bp.route('/recipes/<int:recipe_id>', methods=['GET'])
def get_recipe(recipe_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM recipes WHERE RecipeId = ?', (recipe_id,))
        recipe = cursor.fetchone()
        conn.close()
        if not recipe:
            return jsonify({'error': 'Recipe not found'}), 404
        recipe_dict = dict(recipe)
        array_fields = [
            'RecipeIngredientParts',
            'RecipeInstructions',
            'RecipeIngredientQuantities',
            'Images',
            'Keywords'
        ]
        for field in array_fields:
            recipe_dict[field] = parse_r_vector(recipe_dict.get(field, ''))
        nutrition = {key: recipe_dict.pop(key) for key in [
            'Calories', 'FatContent', 'SaturatedFatContent',
            'CholesterolContent', 'SodiumContent', 'CarbohydrateContent',
            'FiberContent', 'SugarContent', 'ProteinContent'
        ]}
        recipe_dict['nutrition'] = nutrition
        return jsonify(recipe_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500