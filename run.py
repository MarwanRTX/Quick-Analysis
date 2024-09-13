from datetime import timedelta
from app import create_app
from flask_session import Session
import os

app = create_app()

# Set environment to production
os.environ['FLASK_ENV'] = 'production'

# Define folder for uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set the maximum file upload size to 15MB (can be changed based on your requirement)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # 15 MB

# Session configuration (server-side session storage using the filesystem)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False  # Non-permanent session
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Session lifetime of 30 minutes
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session_files')  # Folder for storing session files

# Secret key for session encryption
app.secret_key = 'your_super_secret_key'

# Initialize server-side session
Session(app)

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=False,host='0.0.0.0')
