import pytest
from digit_classifier import app, DigitClassifier, get_classifier
from io import BytesIO
from PIL import Image

@pytest.fixture(scope='module')
def client():
    app.config['TESTING'] = True
    with app.test_client() as testing_client:
        # Initialize classifier
        get_classifier()
        yield testing_client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Digit Classifier" in response.data

def test_predict_valid(client):
    img = Image.new('L', (28, 28), color=255)
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)

    response = client.post(
        '/predict',
        data={'file': (img_io, 'test.png')},
        content_type='multipart/form-data'
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert "digit" in json_data

def test_predict_missing_file(client):
    response = client.post('/predict', data={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "No file uploaded"

def test_predict_invalid_file(client):
    data = {
        'file': (BytesIO(b"invalid data"), 'test.txt')
    }
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 500
    assert "error" in response.get_json()