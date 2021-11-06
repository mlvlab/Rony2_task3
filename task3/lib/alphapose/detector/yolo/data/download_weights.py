import requests
''' Python script which downloads the weights for deepsort from google drive into current folder'''

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    print('Loading ', destination)
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, 'detector/yolo/data/'+destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":

    files = {'1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC': 'yolov3-spp.weights'}

    for google_id in files:
        filename = files[google_id]
        download_file_from_google_drive(google_id, filename)
