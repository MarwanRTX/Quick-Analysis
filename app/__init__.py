from flask import Flask
from .routes import bp as main_bp

def create_app():
    app = Flask(__name__)
    
    # Register the blueprint only once
    app.register_blueprint(main_bp, url_prefix='/')
    
    # Set up other configurations
    app.secret_key = 'your_super_secret_key'
    
    return app
