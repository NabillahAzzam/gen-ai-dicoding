from google.cloud import storage
import os

def download_model_from_gcp(bucket_name, model_dir, local_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=model_dir)

    for blob in blobs:
        if not blob.name.endswith('/'):
            local_path = os.path.join(local_dir, os.path.relpath(blob.name, model_dir))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
