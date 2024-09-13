# forms.py
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired

class UploadForm(FlaskForm):
    file = FileField('Upload CSV', validators=[DataRequired()])
    submit = SubmitField('Upload')
