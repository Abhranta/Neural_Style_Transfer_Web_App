#Here I will be loading the data and transforming the images

import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms as transforms

torch.cuda.empty_cache()


def load_image(image_name):
    image = Image.open(image_name)
    #return image
    image = loader(image).unsqueeze(0)
    return image.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 200

loader = transforms.Compose(
    [
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ]
)


"""
original_img = load_image("/home/abhrant/Pictures/1.jpg")
style_img = load_image("/home/abhrant/Pictures/2.jpg")
"""