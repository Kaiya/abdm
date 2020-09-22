import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

img_url = "https://www.ishn.com/ext/resources/900x550/airplane-plane-flight-900.jpg?1583412590"
# img_url = "https://data.megengine.org.cn/images/cat.jpg"
url, filename = (img_url, "cat.jpg")
# load model
model = models.alexnet(pretrained=True)

# switch to eval mode
model.eval()
# Download an example image from the megengine data website
import urllib
import os
if not os.path.exists('cat.jpg'): 
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

# preprocessing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

def image_loader(image_name):
    image = Image.open(image_name)
    image = preprocessing(image)
    image = torch.unsqueeze(image, 0)
    return image 

image = image_loader("cat.jpg")
out = model(image)
_, index = torch.max(out, 1)
probs = F.softmax(out, dim=1)[0] * 100

with open('imagenet1000_clsidx_to_labels.txt') as f:
    classes = [line.strip() for line in f.readlines()]

print(classes[index[0].long()], ' confidence: ', probs[index[0]].item())
