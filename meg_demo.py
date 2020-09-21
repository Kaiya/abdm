import megengine.hub
model = megengine.hub.load('megengine/models', 'resnet18', pretrained=True)
# or any of these variants
# model = megengine.hub.load('megengine/models', 'resnet34', pretrained=True)
# model = megengine.hub.load('megengine/models', 'resnet50', pretrained=True)
# model = megengine.hub.load('megengine/models', 'resnet101', pretrained=True)
# model = megengine.hub.load('megengine/models', 'resnet152', pretrained=True)
# model = megengine.hub.load('megengine/models', 'resnext50_32x4d', pretrained=True)
model.eval()

# Download an example image from the megengine data website
import urllib
url, filename = ("https://data.megengine.org.cn/images/cat.jpg", "cat.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# Read and pre-process the image
import cv2
import numpy as np
import megengine.data.transform as T
import megengine.functional as F

image = cv2.imread("cat.jpg")
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]),  # BGR
    T.ToMode("CHW"),
])
processed_img = transform.apply(image)[np.newaxis, :]  # CHW -> 1CHW
logits = model(processed_img)
probs = F.softmax(logits)
print(probs)
