import os
from werkzeug.utils import secure_filename

class FileService:
    def __init__(self, upload_folder, download_folder):
        
        self.upload_folder = upload_folder
        self.download_folder = download_folder

    def get_next_sequential_name(self):
        """Gets the next available folder/filename as 001, 002, etc."""
        existing_folders = [int(f) for f in os.listdir(self.download_folder) if f.isdigit()]
        if existing_folders:
            return f"{max(existing_folders) + 1:03d}"
        return "001"

    def save_file_with_sequential_name(self, file):
        """Save the uploaded file and rename it to the next available number."""
        # Get the next sequential folder and filename
        sequential_name = self.get_next_sequential_name()
        
        # Set new filename and folder based on the sequential name
        file_ext = os.path.splitext(file.filename)[1].lower()
        new_filename = f"{sequential_name}{file_ext}"

        # Save the file in the uploads folder
        save_path = os.path.join(self.upload_folder, new_filename)
        file.save(save_path)
        
        # Create a folder in the downloads directory for the file
        download_folder = os.path.join(self.download_folder, sequential_name)
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)

        return new_filename

