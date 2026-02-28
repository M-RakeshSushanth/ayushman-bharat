"""Upload sample_claims.csv to the running FastAPI server using only stdlib."""
import http.client
import mimetypes
import os
import uuid

def upload_file(host, port, path, filepath):
    boundary = uuid.uuid4().hex
    with open(filepath, 'rb') as f:
        file_data = f.read()
    filename = os.path.basename(filepath)
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: text/csv\r\n\r\n'
    ).encode() + file_data + f'\r\n--{boundary}--\r\n'.encode()

    conn = http.client.HTTPConnection(host, port, timeout=60)
    conn.request('POST', path, body=body, headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'Content-Length': str(len(body))
    })
    resp = conn.getresponse()
    result = resp.read().decode()
    print(f'Status: {resp.status}')
    print(f'Response: {result}')
    conn.close()

upload_file('localhost', 8000, '/api/upload', 'sample_claims.csv')
