import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
sys.path.append("/home/whitewalker/Documents/prince/resnet_pipeline")
from ResNet.model import ResNet, _BasicBlock, resnet101



def load_and_infer(image_path, model_checkpoint_path):
    # Define transformations for the input image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Load the model
    model = resnet101(num_classes= 2)  # Replace with your model definition
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)
    
    return predicted_class.item()




image_path = 'path/to/your/image.jpg'  # Replace with the path to your image
# predicted_class = load_and_infer(image_path, "/home/whitewalker/Documents/prince/resnet_pipeline/your_model.pth")
import glob 

image_list = glob.glob("/home/whitewalker/Documents/prince/gender/data/val/male/*.jpg")

# Print the predicted class
female = 0
male = 0
for i in image_list:
    predicted_class = load_and_infer(i, "/home/whitewalker/Documents/prince/resnet_pipeline/ResNet/best.pth")
    if int(predicted_class) == 1:
        male += 1
    else:
        female += 1
print(f"Female : {female}, Male:  {male}")




# print(f"Predicted class index: {predicted_class}")
