from flask import Flask
from auth import auth_bp
from search import search_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(search_bp)

if __name__ == '__main__':
    app.run(debug=True)
