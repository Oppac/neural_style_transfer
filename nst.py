import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms
from PIL import Image

import nst_vgg

def show_img(processed_img):
    unloader = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
        ])
    img = unloader(processed_img.squeeze(0))
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])

    content_img = Image.open("images\jade.jpg")
    content_img = transform(content_img).unsqueeze(0).to(device)
    target_img = torch.randn_like(content_img).requires_grad_(True).to(device)
    #show_img(target_img)

    model = nst_vgg.Vgg19Nst().to(device).eval()
    optimizer = torch.optim.Adam([target_img], lr=0.01)
    mse_loss = torch.nn.MSELoss(reduction='mean')

    for i in range(1, 1000):
        optimizer.zero_grad()
        loss = mse_loss(model(content_img)['conv4_2'], model(target_img)['conv4_2'])
        loss.backward(retain_graph=True)
        optimizer.step()

        if i%50 == 0:
            print(loss)

    show_img(target_img)
