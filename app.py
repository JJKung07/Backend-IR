from flask import Flask
from flask_cors import CORS
from auth import auth_bp
from search import search_bp, register_commands
from recipes import recipes_bp
from user import user_bp

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(search_bp)
app.register_blueprint(recipes_bp)
app.register_blueprint(user_bp)

# Register CLI commands
register_commands(app)

if __name__ == '__main__':
    app.run(debug=True)