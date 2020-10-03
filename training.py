import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision
import numpy as np

from dataloader import *
from model import VGG


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

original_img = load_image("/home/abhrant/Pictures/1.jpg")
style_img = load_image("/home/abhrant/Pictures/3.jpg")

generated = original_img.clone().requires_grad_(True)
model = VGG().to(device).eval()


total_steps = 401
learning_rate = 0.001
alpha = 0.5
beta = 0.5
optimizer = optim.Adam([generated], lr=learning_rate)




for step in range(total_steps):
    # Obtain the convolution features in specifically chosen layers
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    # Loss is 0 initially
    style_loss = original_loss = 0

    # iterate through all the features for the chosen layers
    for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
        # batch_size will just be 1
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)
        # Compute Gram Matrix of generated
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )
        # Compute Gram Matrix of Style
        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        temp = generated
        temp = temp.to("cpu").squeeze(0)
        to_pil_image = transforms.ToPILImage()
        results = to_pil_image(temp)
        #img = Image.show(temp)
        results.save("generated" + str(step) , "png")

torch.save(model.state_dict() , "/home/abhrant/Neural_Style_Transfer_Web_App/nst_weight.pt")