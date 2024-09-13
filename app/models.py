from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from celery import Celery

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
db = SQLAlchemy(app)

# Cache configuration
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Celery configuration
celery = Celery(app.name, broker='amqp://guest@localhost//')

# Data model
class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.Text, nullable=False)

# Increase the maximum allowed size of the session cookie
app.config['PERMANENT_SESSION_LIFETIME'] = 31536000  # 1 year
app.config['SESSION_REFRESH_EACH_REQUEST'] = False

# Route to handle data
@app.route('/data', methods=['POST'])
def handle_data():
    data = request.get_json()
    new_data = Data(data=data)
    db.session.add(new_data)
    db.session.commit()
    celery.send_task('process_data', args=[data])
    return jsonify(success=True, message='Data saved successfully!'), 200

# Route to get data
@app.route('/data', methods=['GET'])
def get_data():
    data = cache.get('data')
    if data is None:
        data = Data.query.all()
        cache.set('data', data)
    return jsonify(success=True, data=[d.data for d in data]), 200