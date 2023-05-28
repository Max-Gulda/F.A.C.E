import os
import base64
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, ClusterTimeoutOptions, QueryOptions, GetOptions
from couchbase.transcoder import JSONTranscoder, RawJSONTranscoder

server_url = 'couchbases://cb.v7t-bay8ity8htrz.cloud.couchbase.com'
bucket_name = 'Main'
accessname = 'FACE'
secret = 'fazneb-3hebma-cicsyD!'

################################################################ UPLOAD ####################################################################

def upload_images():

    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(server_url, ClusterOptions(
            authenticator=PasswordAuthenticator(accessname, secret),
            transcoder=JSONTranscoder()
        ))
        bucket = cluster.bucket(bucket_name)

        # Get the list of subdirectories in the dataset folder
        subdirs = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]

        # Iterate over the subdirectories and upload each one
        for subdir in subdirs:
            folder_path = os.path.join('dataset', subdir)

            # Get the list of JPG files in the folder
            file_names = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]

            # Accumulate the file paths in a list
            file_paths = []
            for file_name in file_names:
                file_path = os.path.join(folder_path, file_name)
                file_paths.append(file_path)
                print(file_path)

            # Create a dictionary to hold the file data
            file_data_dict = {}

            # Read each JPG file and store base64 encoded data in the dictionary
            for file in file_paths:
                with open(file, 'rb') as file_obj:
                    file_data = file_obj.read()
                    file_data_base64 = base64.b64encode(file_data).decode('utf-8')
                    file_name = file.split('/')[-1]
                    file_data_dict[file_name] = file_data_base64

            # Insert the dictionary of file data as a single document
            collection = bucket.collection('user_jpgs')
            document_id = subdir
            collection.upsert(document_id, file_data_dict)

        print('Dataset uploaded successfully.')

    except Exception as e:
        print(f'Error uploading dataset: {e}')

def upload_npz_file():
    file_path = 'Data/face_embeddings_done_4classes.npz'
    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(server_url, ClusterOptions(
            authenticator=PasswordAuthenticator(accessname, secret),
            transcoder=RawJSONTranscoder()
        ))
        bucket = cluster.bucket(bucket_name)
        collection = bucket.collection('npz')

        # Read the binary data from the NPZ file
        with open(file_path, 'rb') as file_obj:
            file_data = file_obj.read()

        # Upload the file data as a single document
        file_name = os.path.basename(file_path)
        collection.upsert(file_name, file_data)

        print(f'File "{file_name}" uploaded successfully.')

    except Exception as e:
        print(f'Error uploading file: {e}')

def upload_pkl_file():
    file_path = 'Data/svm_model_160x160.pkl'
    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(server_url, ClusterOptions(
            authenticator=PasswordAuthenticator(accessname, secret),
            transcoder=RawJSONTranscoder()
        ))
        bucket = cluster.bucket(bucket_name)
        collection = bucket.collection('pkl')

        # Read the binary data from the PKL file
        with open(file_path, 'rb') as file_obj:
            file_data = file_obj.read()

        # Upload the file data as a single document
        file_name = os.path.basename(file_path)
        collection.upsert(file_name, file_data)

        print(f'File "{file_name}" uploaded successfully.')

    except Exception as e:
        print(f'Error uploading file: {e}')

def upload_model():
    upload_npz_file()
    upload_pkl_file()

def upload_token_files():
    dataset_folder = 'dataset'

    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(server_url, ClusterOptions(
            authenticator=PasswordAuthenticator(accessname, secret),
            transcoder=RawJSONTranscoder()
        ))
        bucket = cluster.bucket(bucket_name)
        collection = bucket.collection('calendar')

        for root, dirs, files in os.walk(dataset_folder):
            if "ms_graph_api_token.json" in files:
                file_path = os.path.join(root, "ms_graph_api_token.json")

                # Read the binary data from the token file
                with open(file_path, 'rb') as file_obj:
                    file_data = file_obj.read()

                # Use the subdirectory name as the document ID
                subdir_name = os.path.basename(root)

                # Upload the token file data with the subdirectory name as the document ID
                collection.upsert(subdir_name, file_data)

                print(f'Token file uploaded successfully for subdirectory: {subdir_name}')

    except Exception as e:
        print(f'Error uploading token file: {e}')

def upload_everything():
    upload_token_files()
    upload_model()
    upload_images()

################################################################ DOWNLOAD ###################################################################

def download_images(document_id, output_folder='dataset'):
    try:
       # Connect to the Couchbase cluster
       cluster = Cluster(server_url, ClusterOptions(
           authenticator=PasswordAuthenticator(accessname, secret)
       ))
       bucket = cluster.bucket(bucket_name)
       collection = bucket.collection('user_jpgs')

       # Retrieve the document
       document = collection.get(document_id)

       # Retrieve image data from the document
       file_data_dict = document.content_as[dict]

       # Create a folder for the document if it doesn't exist
       document_folder = os.path.join(output_folder, document_id)
       os.makedirs(document_folder, exist_ok=True)

       # Save each image file from the document
       for file_name, file_data_base64 in file_data_dict.items():
           # Decode base64 data
           file_data = base64.b64decode(file_data_base64)

           # Save the image file
           file_path = os.path.join(document_folder, file_name)
           with open(file_path, 'wb') as file_obj:
               file_obj.write(file_data)

           print(f'Image "{file_name}" downloaded and saved to "{file_path}".')

       print('All images downloaded successfully.')

    except Exception as e:
        print(f'Error downloading images: {e}')

def get_all_documents(collection = 'user_jpgs'):
    cluster = Cluster(server_url, ClusterOptions(
        authenticator=PasswordAuthenticator(accessname, secret)
        ))
    bucket = 'Main'

    # Retrieve all document IDs from the collection
    query = f'SELECT DISTINCT RAW META().id FROM `{bucket}`.`_default`.`{collection}`'
    result = cluster.query(query)
    # Print the document IDs
    list = []
    for row in result:
        list.append(row)

    return list

def download_all_images():
    list = get_all_documents()
    for subdirs in list:
        download_images(subdirs)

def download_npz_file(download_path = 'Data'):
    file_name = 'face_embeddings_done_4classes.npz'
    #try:
    # Connect to the Couchbase cluster
    cluster = Cluster(server_url, ClusterOptions(
        authenticator=PasswordAuthenticator(accessname, secret),
        transcoder=RawJSONTranscoder()
    ))
    bucket = cluster.bucket(bucket_name)
    collection = bucket.collection('npz')

    # Retrieve the file data for the given file name
    result = collection.get(file_name)
    file_data = bytes(result.value)

    # Save the file data to disk
    save_path = os.path.join(download_path, file_name)
    with open(save_path, 'wb') as file_obj:
        file_obj.write(file_data)

    print(f'File "{file_name}" downloaded successfully.')

    #except Exception as e:
    #    print(f'Error downloading file: {e}')

def download_pkl_file(download_path = 'Data'):
    file_name = 'svm_model_160x160.pkl'
    # Connect to the Couchbase cluster
    cluster = Cluster(server_url, ClusterOptions(
        authenticator=PasswordAuthenticator(accessname, secret),
        transcoder=RawJSONTranscoder()
    ))
    bucket = cluster.bucket(bucket_name)
    collection = bucket.collection('pkl')

    # Retrieve the file data for the given file name
    result = collection.get(file_name)
    file_data = bytes(result.value)

    # Save the file data to disk
    save_path = os.path.join(download_path, file_name)
    with open(save_path, 'wb') as file_obj:
        file_obj.write(file_data)

    print(f'File "{file_name}" downloaded successfully.')

def download_model():
    download_npz_file()
    download_pkl_file()

def download_token_file(document_id, output_folder='dataset'):
    #try:
    # Connect to the Couchbase cluster
    # Connect to the Couchbase cluster
    cluster = Cluster(server_url, ClusterOptions(
        authenticator=PasswordAuthenticator(accessname, secret),
        transcoder=RawJSONTranscoder()
    ))
    bucket = cluster.bucket(bucket_name)
    collection = bucket.collection('calendar')

    # Retrieve the token file for the specified document ID
    result = collection.get(document_id)
    file_data = bytes(result.value)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a subdirectory with the document ID as the folder name
    subdirectory = os.path.join(output_folder, document_id)
    os.makedirs(subdirectory, exist_ok=True)

    # Save the token file in the subdirectory
    file_path = os.path.join(subdirectory, "ms_graph_api_token.json")
    with open(file_path, 'wb') as file_obj:
        file_obj.write(file_data)

    print(f'Token file downloaded successfully for document ID: {document_id}')

    #except Exception as e:
    #    print(f'Error downloading token file: {e}')

def download_all_tokens():
    list = get_all_documents(collection = 'calendar')
    for subdir in list:
        download_token_file(subdir)

def download_everything():
    download_all_tokens()
    download_model()
    download_all_images()

################################################################ DELETE #####################################################################

def delete_document(document_id, collection_name='calendar'):
    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(server_url, ClusterOptions(
            authenticator=PasswordAuthenticator(accessname, secret),
            transcoder=RawJSONTranscoder()
        ))
        bucket = cluster.bucket(bucket_name)
        collection = bucket.collection(collection_name)

        # Delete the document by its document ID
        collection.remove(document_id)

        print(f'Document "{document_id}" deleted successfully.')

    except Exception as e:
        print(f'Error deleting document: {e}')

def delete_user(user):
    delete_document(user, collection_name='calendar')
    delete_document(user, collection_name='user_jpgs')

if __name__ == "__main__":
    pass

