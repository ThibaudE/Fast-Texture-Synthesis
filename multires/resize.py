import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as fn
import torch
from PIL import Image
import glob
import os

path = '/home/thomas/Downloads/CompImg/TestData256/'

for file in glob.glob(os.path.join(path, '*.jpg')):
    print(file)
    out_file = file[:-3] + '_256.jpg'
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(256, 256)),
    ])
    # img = T(Image.open('/home/thomas/Downloads/CompImg/TestData/test_image.jpg'))
    img = T(Image.open(file)).float()
    save_image(img, out_file)