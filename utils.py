import os
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

matplotlib.use('Agg')


def image_loader(image_name, imsize):
    # define image preprocess
    pre_process = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor()]
    )

    image = Image.open(image_name)
    
    # fake batch dimension required to fit network's input dimensions
    image = pre_process(image).unsqueeze(0)
    
    return image


def save_results(images: list, titles: list=None, save_name='test.png'):
    unloader = transforms.ToPILImage()  # reconvert into PIL image

    if titles is None:
        titles = ['none' for _ in range(len(images))]

    n_images = len(images)
    assert n_images == len(titles)

    plt.cla()
    fig = plt.figure()
    for idx, (img, tle) in enumerate(zip(images, titles)): 
        img = img.cpu().clone()        # we clone the tensor to not do changes on it
        img = img.squeeze(0)           # remove the fake batch dimension
        img = unloader(img)

        _ax = fig.add_subplot(1, n_images, idx + 1)
        _ax.set_title(tle)

        plt.imshow(img)
        plt.axis('off')

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
    
    os.makedirs('./result_imgs', exist_ok=True)
    plt.savefig('./result_imgs/' + save_name, bbox_inches='tight', pad_inches=0)
    print(f'image saved! --> ./result_imgs/{save_name}')
