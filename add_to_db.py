import couchbase
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
import json
import base64
import os


def init():
    # Set the admin credentials
    admin_username = 'FACE'
    admin_password = 'Flempan123!'

    # Set the Couchbase connection parameters
    cluster_url = 'couchbases://cb.a6cm4sbxf8i5qvak.cloud.couchbase.com'
    cluster_options = ClusterOptions(PasswordAuthenticator(admin_username, admin_password))

    # Connect to the Couchbase cluster
    cluster = Cluster(cluster_url, cluster_options)
    bucket_name = 'travel-sample'
    bucket = cluster.bucket(bucket_name)
    collection = bucket.default_collection()

    # Define the document for svm_model_160x160.pkl
    svm_model_doc = {
        'type': 'svm_model'
    }

    # Save the document for svm_model_160x160.pkl
    svm_model_doc_id = 'svm_model_160x160.pkl'
    collection.upsert(svm_model_doc_id, svm_model_doc)

    # Attach the svm_model_160x160.pkl file to the svm_model_doc
    svm_model_file = "Data/svm_model_160x160.pkl"
    with open(svm_model_file, "rb") as f:
        svm_model_data = f.read()
        encoded_data = base64.b64encode(svm_model_data).decode('utf-8')
        attachments = {
            'name': 'svm_model_160x160.pkl',
            'data': encoded_data,
            'content_type': 'application/octet-stream'
        }
        svm_model_doc['_attachments'] = attachments
        collection.upsert(svm_model_doc_id, svm_model_doc)

    # Define the document for face_embeddings_done_4classes.npz
    face_embeddings_doc = {
        'type': 'face_embeddings'
    }

    # Attach the face_embeddings_done_4classes.npz file to the face_embeddings_doc
    face_embeddings_file = "Data/face_embeddings_done_4classes.npz"
    with open(face_embeddings_file, "rb") as f:
        face_embeddings_data = f.read()
        encoded_data = base64.b64encode(face_embeddings_data).decode('utf-8')
        attachments = {
            'name': 'face_embeddings_done_4classes.npz',
            'data': encoded_data,
            'content_type': 'application/octet-stream'
        }
        face_embeddings_doc['_attachments'] = attachments
        face_embeddings_doc_id = 'face_embeddings_done_4classes.npz'
        collection.upsert(face_embeddings_doc_id, face_embeddings_doc)

    dir_path = 'dataset'
    dir_list = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) and not d.startswith('.')]
    num_dirs = len(dir_list)
    add_to_progressbar = 1.0 / num_dirs
    progressbar_len = 0
    # Iterate over each directory in the dataset directory
    for dirname in os.listdir('dataset'):
        if dirname == '.DS_Store':
            continue
        progressbar_len += add_to_progressbar
        #self.train_progressbar.progressbar.set(progressbar_len)
        # Remove the underscore from the directory name
        fullName = dirname.replace('_', ' ')

        # Define the user data
        user_data = {
            'fullName': fullName,
            'images': []
        }

        if os.path.exists(os.path.join('dataset', dirname, 'ms_graph_api_token.json')):
            with open(os.path.join('dataset', dirname, 'ms_graph_api_token.json'), 'rb') as f:
                print("Found API KEY for " + dirname)
                json_bytes = f.read()

            json_base64 = base64.b64encode(json_bytes).decode('utf-8')

            # Add the attachment to the user data
            attachments = {
                'ms_graph_api_token.json': {
                    'content_type': 'application/json',
                    'data': json_base64,
                    'stub': False  # Set stub to False
                }
            }
            user_data['_attachments'] = attachments
        else:
            print("Found no API KEY for " + dirname)

        # Iterate over each file in the directory
        dir_path = os.path.join('dataset', dirname)
        print(dir_path)
        file_list = [f for f in os.listdir(dir_path) if not f.startswith('.') and f != '.DS_Store' and os.path.isfile(os.path.join(dir_path, f))]

        # Loop through the file list and process each file
        for filename in file_list:
            # Check if the file is a JPEG image
            if filename.lower().endswith('.jpg'):
                # Add the image to the user data
                user_data['images'].append(filename)

                # Read the image file into a bytes object
                with open(os.path.join('dataset', dirname, filename), 'rb') as f:
                    image_bytes = f.read()

                # Encode the bytes as base64
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                # Get the content type of the image
                content_type = 'image/jpeg'

                # Add the attachment to the user data
                attachments = user_data.get('_attachments', {})

                attachments[filename] = {
                    'content_type': content_type,
                    'data': image_base64,
                    'length': len(image_bytes),
                    'stub': False
                }

                user_data['_attachments'] = attachments

        # Convert all string values to UTF-8
        user_data = json.loads(json.dumps(user_data, ensure_ascii=False).encode('utf8'))

        # Create a new document in the bucket
        collection.upsert(str(user_data), user_data)

        # Print the ID and revision of the new document
        print('New document created for', fullName)
    print("Done uploading all users.")


init()
