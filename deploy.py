import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network architecture (must match the trained model)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Define the transform to preprocess the input image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
    transforms.Resize((28, 28)),                  # Resize the image to 28x28
    transforms.ToTensor(),                        # Convert the image to a tensor
    transforms.Normalize((0.5,), (0.5,))          # Normalize the image
])

def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python deploy_model.py <image_path>")
    else:
        image_path = sys.argv[1]
        prediction = predict_image(image_path)
        print(f'Predicted digit: {prediction}')
