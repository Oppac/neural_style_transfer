import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
from torchvision import models, transforms

import nst_vgg

def show_img(processed_img, denorm=True, show=True, save=False, name="img"):
    if denorm:
        unloader = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.ToPILImage()
            ])
    else:
        unloader = transforms.ToPILImage()
    img = unloader(processed_img.squeeze(0))
    if save:
        img.save(f"images\{name}.jpg")
    if show:
        plt.imshow(img)
        plt.show()

def gram_matrix(tensor):
    _, channels, height, width = tensor.size()
    tensor = tensor.view(channels, height * width)
    gram = torch.mm(tensor, tensor.t())
    return gram / (channels * height * width)

if __name__ == "__main__":

    content_img_path = "images\jade.jpg"
    style_img_path = "images\starry_night.jpg"
    output_name = "starry_jade"

    nb_epoch = 30
    content_weights = 1
    style_weights = 10
    random_init = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    content_img = Image.open(content_img_path)
    content_img = transform(content_img).unsqueeze(0).to(device)

    style_img = Image.open(style_img_path)
    style_img = transform(style_img).unsqueeze(0).to(device)

    if random_init:
        output_img = torch.randn_like(content_img).requires_grad_(True).to(device)
    else:
        output_img = content_img.clone().requires_grad_(True).to(device)

    model = nst_vgg.Vgg19Nst().to(device).eval()
    content_maps = model(content_img)
    style_maps = model(style_img)

    content_layers = ['conv4_2']
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    content_rep = {layer: content_maps[layer].squeeze(axis=0) for layer in content_layers}
    style_grams = {layer: gram_matrix(style_maps[layer]) for layer in style_layers}

    optimizer = torch.optim.LBFGS([output_img])
    mse_loss_content = torch.nn.MSELoss(reduction='mean')
    mse_loss_style = torch.nn.MSELoss(reduction='sum')

    run = 0
    while run <= nb_epoch:
        def closure():
            global run
            optimizer.zero_grad()

            content_loss = 0
            style_loss = 0
            output_rep = model(output_img)

            for name in content_layers:
                content_loss += mse_loss_content(content_rep[name], output_rep[name].squeeze(axis=0))
            content_loss /= len(content_layers)

            output_grams = {layer: gram_matrix(output_rep[layer]) for layer in style_layers}
            for name in style_layers:
                style_loss += mse_loss_style(style_grams[name], output_grams[name])
            style_loss /= len(style_layers)

            total_loss = (content_weights * content_loss) + (style_weights * style_loss)
            total_loss.backward()

            if run%5 == 0:
                print(f" {run} | total_loss: {total_loss} | content_loss: {content_weights * content_loss} | style_loss: {style_weights * style_loss}")
            run += 1
            return total_loss

        optimizer.step(closure)

    output_img.data.clamp_(-1.5, 1.5)
    show_img(output_img, save=True, name=output_name)
