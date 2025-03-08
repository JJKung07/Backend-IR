import os
from spellchecker import SpellChecker
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from datetime import datetime, timedelta
import jwt
import bcrypt
import re

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
app.config['SECRET_KEY'] = '56d06939523008f63729b436b6872a7b017b1a6389fbf16f2d98d828471ce5e2'  # Change this in production
app.config['JWT_EXPIRATION_DELTA'] = 24 * 60 * 60  # 24 hours in seconds
DB_PATH = 'resoures/db.db'


# Database setup
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create Users table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        full_name TEXT,
        profile_image_url TEXT,
        is_active BOOLEAN DEFAULT 1
    )
    ''')

    conn.commit()
    conn.close()


# Initialize database on startup
init_db()


# Helper functions
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def validate_email(email):
    """Validate email format"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """Validate password strength"""
    # At least 8 characters, containing at least one digit and one letter
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one number"

    if not any(char.isalpha() for char in password):
        return False, "Password must contain at least one letter"

    return True, "Password is valid"


def generate_token(user_id):
    """Generate JWT token for authenticated user"""
    payload = {
        'exp': datetime.utcnow() + timedelta(seconds=app.config['JWT_EXPIRATION_DELTA']),
        'iat': datetime.utcnow(),
        'sub': user_id
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')


def hash_password(password):
    """Hash a password for storing"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt)


def check_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)


# Routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()

    # Extract user data
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    full_name = data.get('full_name', '').strip()

    # Validate input
    if not username or not email or not password:
        return jsonify({'error': 'Username, email, and password are required'}), 400

    if not validate_email(email):
        return jsonify({'error': 'Invalid email format'}), 400

    is_valid, password_message = validate_password(password)
    if not is_valid:
        return jsonify({'error': password_message}), 400

    # Hash password
    hashed_password = hash_password(password)

    # Store in database
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            'INSERT INTO Users (username, email, password_hash, full_name, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)',
            (username, email, hashed_password, full_name, datetime.now(), datetime.now())
        )
        conn.commit()

        # Get the new user_id
        user_id = cursor.lastrowid

        # Generate token
        token = generate_token(user_id)

        return jsonify({
            'message': 'User registered successfully',
            'user_id': user_id,
            'token': token
        }), 201

    except sqlite3.IntegrityError as e:
        # Handle unique constraint violations
        error_message = str(e)
        if "username" in error_message:
            return jsonify({'error': 'Username already exists'}), 409
        elif "email" in error_message:
            return jsonify({'error': 'Email already exists'}), 409
        else:
            return jsonify({'error': 'Registration failed due to database constraint'}), 409
    finally:
        conn.close()


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    # Extract login data
    login_id = data.get('login_id', '').strip()  # Could be username or email
    password = data.get('password', '')

    if not login_id or not password:
        return jsonify({'error': 'Login ID and password are required'}), 400

    # Check database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Try to find user by username or email
    cursor.execute(
        'SELECT * FROM Users WHERE username = ? OR email = ?',
        (login_id, login_id)
    )

    user = cursor.fetchone()
    conn.close()

    if user is None:
        return jsonify({'error': 'Invalid credentials'}), 401

    # Convert user to dict for easier access
    user_dict = dict(user)

    # Check if user is active
    if not user_dict['is_active']:
        return jsonify({'error': 'Account is deactivated'}), 403

    # Verify password
    stored_password = user_dict['password_hash']
    if check_password(stored_password, password):
        # Generate token
        token = generate_token(user_dict['user_id'])

        return jsonify({
            'message': 'Login successful',
            'user_id': user_dict['user_id'],
            'username': user_dict['username'],
            'email': user_dict['email'],
            'full_name': user_dict['full_name'],
            'token': token
        }), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401


# Middleware to verify token
def token_required(f):
    def decorated(*args, **kwargs):
        token = None

        # Check if token is in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]

        if not token:
            return jsonify({'error': 'Token is missing'}), 401

        try:
            # Verify token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            user_id = data['sub']

            # Get user from database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM Users WHERE user_id = ?', (user_id,))
            user = cursor.fetchone()
            conn.close()

            if user is None:
                return jsonify({'error': 'Invalid token: User not found'}), 401

            # Check if user is active
            if not dict(user)['is_active']:
                return jsonify({'error': 'Account is deactivated'}), 403

        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        return f(user_id, *args, **kwargs)

    decorated.__name__ = f.__name__
    return decorated


# Example protected route
@app.route('/api/user/profile', methods=['GET'])
@token_required
def get_user_profile(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT user_id, username, email, full_name, profile_image_url, created_at FROM Users WHERE user_id = ?',
        (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user is None:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'user_id': user['user_id'],
        'username': user['username'],
        'email': user['email'],
        'full_name': user['full_name'],
        'profile_image_url': user['profile_image_url'],
        'created_at': user['created_at']
    }), 200


def correct_typos(query):
    """Correct typos in the search query"""
    spell = SpellChecker()
    corrected_words = [spell.correction(word) or word for word in query.split()]
    return ' '.join(corrected_words)


@app.route('/api/search', methods=['GET'])
@token_required
def search_recipes(user_id):
    raw_query = request.args.get('query', '').strip().lower()
    search_type = request.args.get('type', 'all')  # all/name/ingredient/process

    # Correct typos
    corrected_query = correct_typos(raw_query)
    if raw_query != corrected_query:
        return jsonify({
            'suggestion': corrected_query,
            'message': 'Did you mean:'
        }), 200

    # Build dynamic SQL based on search type
    conn = get_db_connection()
    cursor = conn.cursor()

    search_terms = []
    conditions = []

    # WHERE clause parameters
    if search_type in ['all', 'name']:
        conditions.append("Name LIKE ?")
        search_terms.append(f"%{corrected_query}%")

    if search_type in ['all', 'ingredient']:
        conditions.append("RecipeIngredientParts LIKE ?")
        search_terms.append(f"%{corrected_query}%")

    if search_type in ['all', 'process']:
        conditions.append("RecipeInstructions LIKE ?")
        search_terms.append(f"%{corrected_query}%")

    # CASE statement parameters (needs 3 placeholders)
    case_params = [f"%{corrected_query}%"] * 3

    # Combine all parameters
    all_params = search_terms + case_params

    # Final SQL query
    sql = f'''
    SELECT *,
        (CASE 
            WHEN Name LIKE ? THEN 3 
            WHEN RecipeIngredientParts LIKE ? THEN 2 
            WHEN RecipeInstructions LIKE ? THEN 1 
            ELSE 0 
        END) AS relevance
    FROM recipes
    {'WHERE ' + ' OR '.join(conditions) if conditions else ''}
    ORDER BY relevance DESC, AggregatedRating DESC
    LIMIT 20
    '''

    try:
        # Execute with combined parameters
        cursor.execute(sql, all_params)
        results = cursor.fetchall()
    finally:
        conn.close()

    # Format results
    recipes = [{
        'recipeId': row['RecipeId'],
        'name': row['Name'],
        'image': row['Images'].split(',')[0] if row['Images'] else None,
        'rating': row['AggregatedRating'],
        'ingredients': row['RecipeIngredientParts'].split(', ') if row['RecipeIngredientParts'] else [],
        'matchType': 'Name' if row['relevance'] == 3 else
        'Ingredients' if row['relevance'] == 2 else
        'Instructions'
    } for row in results]

    return jsonify({'results': recipes}), 200

if __name__ == '__main__':
    app.run(debug=True)