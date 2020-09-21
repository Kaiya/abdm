import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torch.autograd import Variable

# load model
model = models.alexnet(pretrained=True)

# switch to eval mode
model.eval()
# Download an example image from the megengine data website
import urllib
url, filename = ("https://data.megengine.org.cn/images/cat.jpg", "cat.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# preprocessing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocessing = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    #image = preprocessing(image).float()
    #image = Variable(image, requires_grad=True)
    #image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    image = preprocessing(image)
    image = torch.unsqueeze(image, 0)
    return image  #assumes that you're using GPU

# image = cv2.imread("cat.jpg")
image = image_loader("cat.jpg")
#processed_img = transforms.apply(image)
out = model(image)
_, index = torch.max(out, 1)
probs = F.softmax(out, dim=1)[0] * 100

with open('imagenet1000_clsidx_to_labels.txt') as f:
    classes = [line.strip() for line in f.readlines()]

print(classes[index[0].long()], ' percent: ', probs[index[0]].item())
