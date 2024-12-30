import pytest
from app import app
from io import BytesIO
from PIL import Image

# Test client for Flask
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Test the home page
def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Digit Classifier" in response.data

# Test prediction with valid input
def test_predict_valid(client):
    # Create a dummy image (28x28 grayscale)
    img = Image.new('L', (28, 28), color=255)  # White image
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)

    # Simulate file upload
    data = {
        'file': (img_io, '7.png')
    }
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    json_data = response.get_json()
    assert "digit" in json_data  # Ensure the response contains a digit

# Test prediction with missing file
def test_predict_missing_file(client):
    response = client.post('/predict', data={}, content_type='multipart/form-data')
    assert response.status_code == 400
    json_data = response.get_json()
    assert "error" in json_data
    assert json_data["error"] == "No file uploaded"

# Test prediction with invalid file
def test_predict_invalid_file(client):
    # Simulate file upload with invalid content
    data = {
        'file': (BytesIO(b"invalid data"), 'test.txt')
    }
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 500
    json_data = response.get_json()
    assert "error" in json_data
