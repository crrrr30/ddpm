import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt

from convnext import *

device = "mps"

idx = 2795
img = plt.imread(f"/Users/jonathan/Downloads/celeba_hq_256/{idx:05}.jpg")
img = torch.Tensor(img).to(device) / 255 * 2 - 1
img = img[None, ...].permute(0, 3, 1, 2)

with torch.no_grad():
    model = ConvNeXt(**tiny).to(device)
    model.load_state_dict(
        torch.load("019.pt", map_location=device)["model"]
    )
    corrupted = img + .5 * torch.randn_like(img)
    out = model(corrupted)
    
    inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)
    inception.fc = nn.Identity()
    inception.eval()
    i = inception(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img))
    o = inception(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(out))
    print(f"Inception Distance: {F.l1_loss(i, o):.6f}")

    img = img[0].permute(1, 2, 0)
    out = out[0].permute(1, 2, 0)
    plt.imshow((torch.cat([img, out], axis=1).clamp_(-1., 1.).cpu().numpy() + 1) / 2)
    plt.show()

