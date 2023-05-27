import os
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.transcoder import RawJSONTranscoder, JSONTranscoder
import base64

# Couchbase connection parameters
server_url = 'couchbases://cb.a6cm4sbxf8i5qvak.cloud.couchbase.com'
bucket_name = 'Main'
accessname = 'FACE'
secret = 'Flempan123!'

def upload(files, document_id):
    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(server_url, ClusterOptions(
            authenticator=PasswordAuthenticator(accessname, secret),
            transcoder=JSONTranscoder()
        ))
        bucket = cluster.bucket(bucket_name)
        collection = bucket.collection('User_jpgs')

        # Create a dictionary to hold the file data
        file_data_dict = {}

        # Read each JPG file and store base64 encoded data in the dictionary
        for file in files:
            with open(file, 'rb') as file_obj:
                file_data = file_obj.read()
                file_data_base64 = base64.b64encode(file_data).decode('utf-8')
                file_name = file.split('/')[-1]
                file_data_dict[file_name] = file_data_base64

        # Insert the dictionary of file data as a single document
        collection.upsert(document_id, file_data_dict)

        print('Document uploaded successfully.')

    except Exception as e:
        print(f'Error uploading document: {e}')

def upload_folder(folder_path, document_prefix):
    try:
        # Get the list of JPG files in the folder
        file_names = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]

        # Accumulate the file paths in a list
        file_paths = []
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            file_paths.append(file_path)
            print(file_path)

        # Upload the list of files as a single document
        document_id = document_prefix
        upload(file_paths, document_id)

    except Exception as e:
        print(f'Error uploading folder: {e}')

def upload_dataset(dataset_path):
    try:
        # Get the list of subdirectories in the dataset folder
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        # Iterate over the subdirectories and upload each one
        for subdir in subdirs:
            folder_path = os.path.join(dataset_path, subdir)
            upload_folder(folder_path, subdir)

    except Exception as e:
        print(f'Error uploading dataset: {e}')


def upload_npz():
    file_path = 'Data/face_embeddings_done_4classes.npz'
    collection_name = 'NPZ'
    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(server_url, ClusterOptions(
            authenticator=PasswordAuthenticator(accessname, secret),
            transcoder=RawJSONTranscoder()
        ))
        bucket = cluster.bucket(bucket_name)
        collection = bucket.collection(collection_name)

        # Read the NPZ file
        with open(file_path, 'rb') as file_obj:
            file_data = file_obj.read()

        # Insert the NPZ file as a document
        collection.upsert(file_path, file_data)

        print('NPZ file uploaded successfully.')

    except Exception as e:
        print(f'Error uploading NPZ file: {e}')

def upload_pkl():
    file_path = 'Data/svm_model_160x160.pkl'
    collection_name = 'PKL'
    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(server_url, ClusterOptions(
            authenticator=PasswordAuthenticator(accessname, secret),
            transcoder=RawJSONTranscoder()
        ))
        bucket = cluster.bucket(bucket_name)
        collection = bucket.collection(collection_name)

        # Read the NPZ file
        with open(file_path, 'rb') as file_obj:
            file_data = file_obj.read()

        # Insert the NPZ file as a document
        collection.upsert(file_path, file_data)

        print('PKL file uploaded successfully.')

    except Exception as e:
        print(f'Error uploading PKL file: {e}')

def download_npz():
    document_id = 'Data/face_embeddings_done_4classes.npz'
    collection_name = 'NPZ'
    output_folder = 'Data'
    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(server_url, ClusterOptions(
            authenticator=PasswordAuthenticator(accessname, secret)
        ))
        bucket = cluster.bucket(bucket_name)
        collection = bucket.collection(collection_name)

        # Retrieve the NPZ file from the collection
        result = collection.get(document_id)
        file_data = result.value

        # Save the NPZ file to the output folder
        output_path = os.path.join(output_folder, os.path.basename(document_id))
        with open(output_path, 'wb') as file_obj:
            file_obj.write(file_data)

        print('NPZ file downloaded successfully.')

    except Exception as e:
        print(f'Error downloading NPZ file: {e}')





def upload_neural_network():
    upload_npz()
    upload_pkl()

download_npz()

#upload_neural_network()

# Dataset path and collection prefix
#dataset_path = 'dataset'

# Call the function to upload each subdirectory as a separate document
#upload_dataset(dataset_path)

