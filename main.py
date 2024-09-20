'''
highly reference to https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
'''
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

from loss import ContentLoss, StyleLoss
from utils import image_loader, save_results


# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        dev = img.device
        self.mean = self.mean.to(dev)
        self.std = self.std.to(dev)
        return (img - self.mean) / self.std


def build_style_model_and_loss_layer(cnn, normalization_mean, normalization_std,
                               style_img, content_img, device,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    '''
    content_layers, style_layers:
        desired depth layers to compute style/content losses :
    '''

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle losses
    content_loss_layers = []
    style_loss_layers = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:  # default: ['conv_4']
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_loss_layers.append(content_loss)

        if name in style_layers:  # default: ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_loss_layers.append(style_loss)
    
    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_loss_layers, content_loss_layers


def run_style_transfer(cnn, normalization_mean, normalization_std, device,
                       content_img, style_img, input_img, num_steps=100,
                       style_weight=1000000, content_weight=1):
    
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_loss_layers, content_loss_layers = build_style_model_and_loss_layer(
        cnn=cnn,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        style_img=style_img,
        content_img=content_img,
        device=device
    )

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])

    print('=== Optimizing the input image... ===')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_loss_layers:
                style_score += sl.loss
            for cl in content_loss_layers:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 20 == 0:
                print(f"run {run}:")
                print(f'\tStyle Loss : {style_score.item():.4f} Content Loss: {content_score.item():.4f}')

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


if __name__ == "__main__":
    ''' setting up config '''
    STYLE_IMG_PATH = './style_imgs/picasso.jpg'
    CONTENT_IMG_PATH = './content_imgs/dancing.jpg'
    SAVE_NAME = 'result_test.jpg'
    
    NUM_STEPS = 150
    STYLE_WEIGHT = 1000000
    CONTENT_WEIGHT = 1
    ''' setting end '''


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # desired size of the output image, use small size if no gpu
    imsize = 512 if torch.cuda.is_available() else 128  

    # loader preprocess input image and turn into tensor
    style_img = image_loader(STYLE_IMG_PATH, imsize).to(device, torch.float)
    content_img = image_loader(CONTENT_IMG_PATH, imsize).to(device, torch.float)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # only need feature extracting block
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    input_img = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)

    output = run_style_transfer(cnn=cnn,
                                normalization_mean=[0.485, 0.456, 0.406],
                                normalization_std=[0.229, 0.224, 0.225],
                                content_img=content_img,
                                style_img=style_img,
                                input_img=input_img,
                                num_steps=NUM_STEPS,
                                style_weight=STYLE_WEIGHT,
                                content_weight=CONTENT_WEIGHT,
                                device=device)

    save_results(images=[style_img, content_img, output],
                 titles=['style', 'content', 'output'],
                 save_name=SAVE_NAME)
