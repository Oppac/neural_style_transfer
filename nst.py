import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from PIL import Image

def show_img(processed_img):
    unloader = transforms.Compose([
        #transforms.Normalize(
            #mean=[-0.485/0.229, -0.456/0.224 -0.406/0.255],
            #std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
        ])
    img = unloader(processed_img)
    plt.imshow(img)
    plt.show()

vgg19 = models.vgg19(pretrained=True)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])

img = Image.open("images\jade.jpg")
img = transform(img)
show_img(img)
#show_img(img)
