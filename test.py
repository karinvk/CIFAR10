import torch
import torchvision
from PIL import Image
from torch import nn

#####resize to input size
test_image_path = "../imgs/test.png"
image = Image.open(test_image_path)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()]) #resize, compose to combine transforms
image = transform(image)
#print(image.shape)
image = torch.reshape(image, (1, 3, 32, 32)) #3 to 4 dim, batch size

#####load model and output class
model = torch.load(model_save_path).to(device)
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))