import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from flask import Flask, request, render_template, jsonify
from PIL import Image
import os

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class DigitClassifier:
    def __init__(self, model_path='mnist_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = SimpleNN().to(self.device)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"Loaded existing model from {model_path}")
        else:
            print("No existing model found. Training new model...")
            self.train_model()
            
        self.model.eval()

    def train_model(self):
        # Training code remains the same
        batch_size = 64
        epochs = 5
        learning_rate = 0.001

        train_dataset = datasets.MNIST(root='./data', train=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                     ]), 
                                     download=True)
        
        test_dataset = datasets.MNIST(root='./data', train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                    ]),
                                    download=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

            self.model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
            print(f'Test set: Average loss: {test_loss:.4f}, '
                  f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def predict(self, image):
        try:
            img = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(img)
                predicted = torch.argmax(output, dim=1).item()
            return predicted
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise e

# Flask application
app = Flask(__name__)

# Create global classifier instance
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = DigitClassifier()
    return _classifier

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Digit Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { text-align: center; }
            .upload-form { margin: 20px 0; padding: 20px; border: 1px solid #ccc; border-radius: 5px; }
            .result { margin-top: 20px; font-size: 1.2em; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Digit Classifier</h1>
            <div class="upload-form">
                <h2>Upload a Digit Image</h2>
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*" required>
                    <button type="submit">Predict</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        classifier = get_classifier()
        img = Image.open(file).convert('L')
        predicted_digit = classifier.predict(img)
        return jsonify({"digit": predicted_digit})
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    _classifier = DigitClassifier()
    app.run(debug=True)