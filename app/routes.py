from datetime import timedelta
from io import StringIO
import os
import shutil
import time
import traceback
import uuid
from venv import logger
from flask import Blueprint, render_template, request, redirect, flash, send_file, session, jsonify, url_for
import numpy as np
import pandas as pd
import logging
import patoolib
from shapely.geometry import shape
from app.utils import DataProcessor
from app.services import FileService

bp = Blueprint('main', __name__)


FILE_EXPIRATION_MINUTES = 1
FILE_EXPIRATION_SECONDS = FILE_EXPIRATION_MINUTES * 60  # 1 minute in seco

def delete_old_files_and_folders(upload_folder, download_folder):

    current_time = time.time()

    # Iterate over files and folders in the upload and download directories
    for folder in [upload_folder, download_folder]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                # Check the last modification time of the file/folder
                if os.path.exists(file_path):
                    last_modified_time = os.path.getmtime(file_path)
                    # If the file/folder is older than the expiration time, delete it
                    if current_time - last_modified_time > FILE_EXPIRATION_SECONDS:
                        if os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        else:
                            os.remove(file_path)
                        print(f"Deleted old file/folder: {file_path}")


def extract_info(df):
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

def get_columns_info(df):
    try :
    
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        null_columns = {col: df[col].isnull().sum() for col in df.columns if df[col].isnull().any()}
        return numeric_cols, object_cols, null_columns
    except Exception as e:
            print(f"An error occurred while saving the plots: {str(e)}")

@bp.route('/')
def index():
    return render_template('index.html')

from flask import jsonify

# Define the error handler for 413 Request Entity Too Large
@bp.route('/upload', methods=['POST'])
def upload_file():
    try:
        session.permanent = True
        upload_folder = 'app/uploads'
        download_folder = 'app/static/downloads'

        delete_old_files_and_folders(upload_folder, download_folder)
        
        if 'file' not in request.files:
            return jsonify(success=False, message='No file part'), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify(success=False, message='No selected file'), 400

        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())  # Generate a unique user ID

        file_service = FileService(upload_folder=upload_folder, download_folder=download_folder)
        
        # Save the file with a sequential name and store the file path
        new_filename = file_service.save_file_with_sequential_name(file)
        file_path = os.path.join('app', 'uploads', new_filename)

        # Derive folder base name and store file path
        file_base_name = os.path.splitext(new_filename)[0]
        folder_path = os.path.join('app', 'static', 'downloads', file_base_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        session['filename'] = new_filename
        session['uploaded_file_path'] = file_path
        session['folder_path'] = folder_path

        if not os.path.exists(file_path):
            return jsonify(success=False, message='File was not saved correctly'), 500

        file_ext = os.path.splitext(new_filename)[1].lower()

        if file_ext == '.csv':
            csv_file = pd.read_csv(file_path)
        elif file_ext == '.tsv':
            csv_file = pd.read_csv(file_path, delimiter='\t')
        elif file_ext in ['.xls', '.xlsx']:
            csv_file = pd.read_excel(file_path)
        else:
            return jsonify(success=False, message='Unsupported file format'), 400

        if csv_file.empty:
            return jsonify(success=False, message='The file is empty or could not be read properly'), 400

        numeric_columns = csv_file.select_dtypes(include=['number', 'datetime'])
        if numeric_columns.empty:
            # Skip plotting if no numeric or datetime columns are found
            session['data_preview'] = {
                'head': csv_file.head().to_html(),
                'tail': csv_file.tail().to_html(),
                'sample': csv_file.sample(n=min(len(csv_file), 5)).to_html(),
                'info': extract_info(csv_file),
                'description': csv_file.describe(include='all').to_html(),
                'formatted_nulls': "\n".join([f"{col}: {num_nulls}" for col, num_nulls in get_columns_info(csv_file)[2].items()]),
                'num_nulls': {col: int(num_nulls) for col, num_nulls in get_columns_info(csv_file)[2].items()},
                'numeric_cols': get_columns_info(csv_file)[0],
                'object_cols': get_columns_info(csv_file)[1],
            }
            session['outliers_data'] = "No outliers detected."
            return jsonify(success=True, message='File uploaded successfully!'), 200

        processor = DataProcessor(csv_file)
        processor.plot_null_patterns(file_base_name)
        processor.generate_plots(file_base_name)

        outliers = processor.detect_outliers()
        session['outliers_data'] = outliers.to_html() if not outliers.empty else "No outliers detected."

        numeric_cols, object_cols, null_columns = get_columns_info(csv_file)
        num_rows = len(csv_file)
        sample_size = min(num_rows, 5)

        if num_rows == 0:
            raise ValueError("DataFrame is empty, cannot sample.")

        sample_df = csv_file.sample(n=sample_size, replace=False)

        def convert_data(x):
            if isinstance(x, (pd.Int64Dtype, pd.Float64Dtype)):
                return int(x) if pd.api.types.is_integer_dtype(x) else float(x)
            return x

        data_preview = {
            'head': csv_file.head().applymap(convert_data).to_html(),
            'tail': csv_file.tail().applymap(convert_data).to_html(),
            'sample': sample_df.applymap(convert_data).to_html(),
            'info': extract_info(csv_file),
            'description': csv_file.describe().applymap(convert_data).to_html(),
            'formatted_nulls': "\n".join([f"{col}: {num_nulls}" for col, num_nulls in null_columns.items()]),
            'num_nulls': {col: int(num_nulls) for col, num_nulls in null_columns.items()},
            'numeric_cols': numeric_cols,
            'object_cols': object_cols,
        }
        session['data_preview'] = data_preview

        return jsonify(success=True, message='File uploaded successfully!'), 200

    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        logger.debug(f"Stack Trace: {traceback.format_exc()}")
        return jsonify(success=False, message=f"An error occurred: {str(e)}"), 500

 


@bp.route('/data_preview', methods=['POST'])
def data_preview():
    try:
        session.permanent = True
        page = int(request.form.get('page', 1))
        per_page = 10
        start = (page - 1) * per_page
        end = start + per_page

        # Load the CSV file from the stored path
        file_path = session.get('data_preview_path')
        if not file_path:
            return jsonify(success=False, message='No data available'), 400

        csv_file = pd.read_csv(file_path)
        data_slice = csv_file.iloc[start:end]
        return jsonify(success=True, data=data_slice.to_html(), page=page)

    except Exception as e:
        return jsonify(success=False, message=f'An error occurred: {str(e)}'), 500


@bp.route('/apply_changes', methods=['POST'])
def apply_changes():
    try:
        file_path = session.get('uploaded_file_path')
        filename = session.get('filename')
        file_base_name = os.path.splitext(filename)[0]
        remove_outliers = request.form.get('remove-outliers') == 'on'
        
        if not filename:
            return jsonify(success=False, message="No file uploaded or filename is missing in session"), 400

        if not file_path:
            return jsonify(success=False, message="No file uploaded"), 400

        # Read the file into a DataFrame
        df = pd.read_csv(file_path)

        # Detect columns with null values and their data types
        column_types = df.dtypes.apply(lambda x: 'object' if x == 'object' else 'numeric').to_dict()

        # Handle null values based on user input
        for col in df.columns:
            action = request.form.get(f'action_{col}')
            specific_value = request.form.get(f'specific_value_{col}')

            if action == 'drop':
                df = df.dropna(subset=[col])
            elif action == 'replace_mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif action == 'replace_median':
                df[col].fillna(df[col].median(), inplace=True)
            elif action == 'replace_specific':
                if specific_value:
                    df.loc[df[col].isnull(), col] = specific_value
        for col in df.columns:
            pattern = request.form.get(f'pattern_{col}')
            if pattern:
                df[col] = df[col].astype(str).replace(pattern, '', regex=True)

        processor = DataProcessor(df)
        processor.plot_null_patterns(file_base_name)

        # Apply outlier removal if selected
        if remove_outliers:
            df = processor.remove_outliers()

        # Save the updated DataFrame to a temporary CSV file
        updated_filename = f"{file_base_name}_updated.csv"
        temp_csv_path = os.path.join('app', 'static', 'downloads', file_base_name, updated_filename)
        df.to_csv(temp_csv_path, index=False)

        # Create a .rar file with only the updated CSV file
        rar_file_path = os.path.join('app', 'static', 'downloads', file_base_name, f"{file_base_name}_files.rar")
        patoolib.create_archive(rar_file_path, [temp_csv_path])

        # Provide the download link to the .rar file
        download_link = url_for('static', filename=f'downloads/{file_base_name}/{file_base_name}_files.rar')

        flash('Changes and pattern removal applied successfully')
        return render_template(
            'datacleaning.html',
            file_base_name=file_base_name,
            num_nulls=df.isnull().sum().sum(),
            num_duplicated=df.duplicated().sum(),
            null_columns={col: df[col].isnull().sum() for col in df.columns if df[col].isnull().any()},
            column_types=column_types,
            all_cols=df.columns.tolist(),
            download_link=download_link  # Pass the download link to the template
        )
    except Exception as e:
        logging.error(f"Error applying changes: {e}")
        return jsonify(success=False, message=str(e)), 500


@bp.route('/dataexplore')
def show_dataexplore():
    filename = session.get('filename', 'No file uploaded')
    data_preview = session.get('data_preview', {})
    outliers_data = session.get('outliers_data', "No data to display.")
    file_base_name = os.path.splitext(filename)[0]

    # Ensure the PDF filename is correctly handled
    pdf_filename = f"static/downloads/{file_base_name}/plots/{file_base_name}_combined.pdf"

    return render_template('dataexplore.html',
                            filename=filename,
                              data_preview=data_preview,
                                text=outliers_data, pdf_filename=pdf_filename,file_base_name=file_base_name)

@bp.route('/datacleaning')
def show_datacleaning():
    # Get the uploaded filename from the session
    filename = session.get('filename', 'No file uploaded')

    if filename == 'No file uploaded':
        return jsonify(success=False, message='No file uploaded'), 400

    file_base_name = os.path.splitext(filename)[0]
    absolute_file_path = os.path.abspath(os.path.join('app', 'uploads', filename))

    if not os.path.exists(absolute_file_path):
        return jsonify(success=False, message='File was not saved correctly'), 500

    try:
        # Read the CSV file
        df = pd.read_csv(absolute_file_path)
    except Exception as e:
        logging.error(f"Error reading the CSV file: {e}")
        return jsonify(success=False, message=f"Error reading the file: {str(e)}"), 500

    # Detect null columns and other statistics
    null_columns = {col: df[col].isnull().sum() for col in df.columns if df[col].isnull().any()}
    num_nulls = sum(null_columns.values())
    num_duplicated = df.duplicated().sum()

    # Update column types for rendering
    column_types = df.dtypes.apply(lambda x: 'object' if x == 'object' else 'numeric').to_dict()
    all_cols = df.columns.tolist()
    # Path to the seaborn null image in the downloads folder
    seaborn_null_image = f'static/downloads/{file_base_name}/plots/{file_base_name}_seaborn_null.png'
    missingno_null_image = f'static/downloads/{file_base_name}/plots/{file_base_name}_missingno_null.png'

    return render_template(
        'datacleaning.html',
        num_nulls=num_nulls,
        num_duplicated=num_duplicated,
        null_columns=null_columns,
        seaborn_null_image=seaborn_null_image,  
        missingno_null_image=missingno_null_image, 
        filename=filename,
        file_base_name=file_base_name,
        column_types=column_types,
        all_cols=all_cols
    )
@bp.route('/download_plots', methods=['GET'])
def download_plots():
    try:
        # Get filename and base folder name
        filename = session.get('filename')
        if not filename:
            return jsonify(success=False, message="No file uploaded"), 400
        
        file_base_name = os.path.splitext(filename)[0]
        
        # Path to the plots directory
        plots_dir = os.path.join('app', 'static', 'downloads', file_base_name, 'plots')

        # Verify the plots directory exists
        if not os.path.exists(plots_dir):
            return jsonify(success=False, message="No plots generated or directory does not exist"), 404

        # Create a new .rar file with only the plot files
        rar_file_path = os.path.join('app', 'static', 'downloads', file_base_name, f"{file_base_name}_plots_only.rar")

        # Get all files in the plots directory
        plot_files = [os.path.join(plots_dir, f) for f in os.listdir(plots_dir) if os.path.isfile(os.path.join(plots_dir, f))]
        
        # If there are no plot files, return an error
        if not plot_files:
            return jsonify(success=False, message="No plots found"), 404

        # Create .rar with the plots
        patoolib.create_archive(rar_file_path, plot_files)

        # Redirect the user to the .rar file to trigger the download
        return redirect(url_for('static', filename=f'downloads/{file_base_name}/{file_base_name}_plots_only.rar'))

    except Exception as e:
        logging.error(f"Error while creating plot archive: {e}")
        return jsonify(success=False, message=f"An error occurred: {e}"), 500
