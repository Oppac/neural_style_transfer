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

def content_loss(target_features, content_features):
    return torch.nn.MSELoss(reduction='mean')(target_features, content_features)

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
    target_img = torch.randn(content_img.data.size()).to(device)
    target_img = torch.autograd.Variable(target_img, requires_grad=True)
    #show_img(target_img)

    model = nst_vgg.Vgg19Nst().to(device).eval()
    content_rep = model(content_img).squeeze(axis=0)
    optimizer = torch.optim.LBFGS([target_img], max_iter=100)

    def closure():
        optimizer.zero_grad()
        target_rep = model(target_img).squeeze(axis=0)
        loss = content_loss(target_rep, content_rep)
        loss.backward()
        print(model.weight.grad())
        return loss

    optimizer.step(closure)

    show_img(target_img)

    #print(content_loss(content_f, target_f))
