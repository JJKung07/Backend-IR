from flask import Flask
from flask_cors import CORS
from auth import auth_bp
from search import search_bp
from recipes import recipes_bp

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Secure CORS configuration

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(search_bp)
app.register_blueprint(recipes_bp)

if __name__ == '__main__':
    app.run(debug=True)