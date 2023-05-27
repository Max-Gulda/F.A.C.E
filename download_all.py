import couchdb
import os
import base64
import json


def init(self):
    # Set the admin credentials
    admin_username = 'admin'
    admin_password = 'f.a.c.e.flempan'

    # Set the CouchDB URL
    couchdb_url = 'http://167.172.105.85:5984'

    # Connect to the CouchDB server
    server = couchdb.Server(couchdb_url)
    server.resource.credentials = (admin_username, admin_password)

    # Open the face database
    db_name = 'face'
    if db_name in server:
        db = server[db_name]
    else:
        print(f"The '{db_name}' database does not exist.")
        exit()

    # Iterate over all documents in the database
    for doc_id in db:
        doc = db[doc_id]

        if 'type' in doc:
            # Create the local folder
            if not os.path.exists("Data"):
                os.makedirs("Data")
            for attachment_name in doc['_attachments']:
                attachment_info = doc['_attachments'][attachment_name]
                if attachment_info.get('stub'):
                    # If the attachment is a stub, fetch the full data
                    attachment_data = db.get_attachment(doc, attachment_name)
                    attachment_info['data'] = attachment_data.read() 
                # Save other attachments to a file
                with open(os.path.join("Data", attachment_name), 'wb') as f:
                    f.write(attachment_info['data'])	
                
            print("Training downloaded")
            
        elif 'fullName' in doc:
            # Create the local folder
            folder_name = f"{doc['fullName'].replace(' ', '_')}"
            if not os.path.exists("dataset/"+folder_name):
                os.makedirs("dataset/"+folder_name)

            # Download the attachments and images
            for attachment_name in doc['_attachments']:
                attachment_info = doc['_attachments'][attachment_name]
                if attachment_info.get('stub'):
                    # If the attachment is a stub, fetch the full data
                    attachment_data = db.get_attachment(doc, attachment_name)
                    attachment_info['data'] = attachment_data.read()
                if attachment_name.lower().endswith('.jpg'):
                    # Save the image to a file
                    with open(os.path.join("dataset/"+folder_name, attachment_name), 'wb') as f:
                        # Encode the image data as base64 and write to file
                        image_base64 = base64.b64encode(attachment_info['data']).decode('utf-8')
                        f.write(base64.b64decode(image_base64))
                else:
                    # Save other attachments to a file
                    with open(os.path.join("dataset/"+folder_name, attachment_name), 'wb') as f:
                        f.write(attachment_info['data'])

            # Print a message to indicate success
            print(f"Data for '{doc['fullName']}' has been downloaded to '{'dataset/'+folder_name}'.")
            
        
        else:
            print(f"No required field found for document {doc_id}.")
