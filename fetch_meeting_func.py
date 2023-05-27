import webbrowser
import json
import os
import msal
import requests
from datetime import datetime, timedelta

GRAPH_API_ENDPOINT = 'https://graph.microsoft.com/v1.0'

def generate_access_token(path):
    #app_id = '3cff8cea-21c6-4547-8620-7a809a0615f1'
    app_id = 'c4614fe4-b441-48bc-af4e-211f00c1b27e'
    scopes = ['Calendars.Read']

    filename = 'ms_graph_api_token.json'
    filepath = os.path.join(path, filename)

    # Save Session Token as a token file
    access_token_cache = msal.SerializableTokenCache()

            # Check if the token file exists in the path
    if not os.path.exists(filepath):
        print('Token file not found')
        return None

    # read the token file
    access_token_cache.deserialize(open(filepath, "r").read())

    # assign a SerializableTokenCache object to the client instance
    client = msal.PublicClientApplication(client_id=app_id, token_cache=access_token_cache)

    accounts = client.get_accounts()
    if accounts:
        # load the session
        token_response = client.acquire_token_silent(scopes, accounts[0])
        if 'access_token' in token_response:
            return token_response

    # authenticate your account as usual
    flow = client.initiate_device_flow(scopes=scopes)
    if 'user_code' in flow:
        print('user_code: ' + flow['user_code'])
        webbrowser.open('https://microsoft.com/devicelogin')
        token_response = client.acquire_token_by_device_flow(flow)
    else:
        print(f"Device flow authentication failed: {flow.get('error')}")
        token_response = None

    if token_response:
        with open(filepath, 'w') as _f:
            _f.write(access_token_cache.serialize())

    return token_response

def fetch_next_meeting_time(json_file_path):
    # Step 1. Generate Access Token
    access_token = generate_access_token(json_file_path)

    if access_token == None:
        return None

    if access_token:
        headers = {
            'Authorization': 'Bearer ' + access_token['access_token']
        }

        # Step 2. Get the next meeting
        response = requests.get(
            GRAPH_API_ENDPOINT + '/me/calendarview',
            headers=headers,
            params={
                'startDateTime': datetime.utcnow().isoformat(),
                'endDateTime': (datetime.utcnow() + timedelta(days=1)).isoformat(),
                '$orderby': 'start/dateTime',
                '$top': '1'
            }
        )

        if response.status_code == 200:
            events = response.json()['value']
            if len(events) > 0:
                next_meeting = events[0]
                start_time = (datetime.fromisoformat(next_meeting['start']['dateTime'][:-8]) + timedelta(hours=2)).strftime("%H:%M")
                #next_meeting_time = int(start_time.replace(":", ""))
                return start_time
            else:
                return None
        else:
            print(f"Error fetching next meeting: {response.status_code} - {response.reason}")
            return None
    else:
        print("Failed to generate access token.")
        return None

def create_json_token(path):

    #app_id = '3cff8cea-21c6-4547-8620-7a809a0615f1'
    app_id = 'c4614fe4-b441-48bc-af4e-211f00c1b27e'
    scopes = ['Calendars.Read']

    filename = 'ms_graph_api_token.json'
    filepath = os.path.join(path, filename)

    # Save Session Token as a token file
    access_token_cache = msal.SerializableTokenCache()

    # read the token file
    if os.path.exists(filepath):
        access_token_cache.deserialize(open(filepath, "r").read())
        token_detail = json.load(open(filepath,))
        token_detail_key = list(token_detail['AccessToken'].keys())[0]
        token_expiration = datetime.fromtimestamp(int(token_detail['AccessToken'][token_detail_key]['expires_on']))
        if datetime.now() > token_expiration:
            os.remove(filepath)
            access_token_cache = msal.SerializableTokenCache()

    # assign a SerializableTokenCache object to the client instance
    client = msal.PublicClientApplication(client_id=app_id, token_cache=access_token_cache)

    accounts = client.get_accounts()
    if accounts:
        # load the session
        token_response = client.acquire_token_silent(scopes, accounts[0])
    else:
        # authenticate your account as usual
        flow = client.initiate_device_flow(scopes=scopes)
        if 'user_code' in flow:
            print('user_code: ' + flow['user_code'])
            webbrowser.open('https://microsoft.com/devicelogin')
            token_response = client.acquire_token_by_device_flow(flow)
        else:
            print(f"Device flow authentication failed: {flow.get('error')}")
            token_response = None

    if token_response:
        with open(filepath, 'w') as _f:
            _f.write(access_token_cache.serialize())

    return None

if __name__ == "__main__":
    pass
