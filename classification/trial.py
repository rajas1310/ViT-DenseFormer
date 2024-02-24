from vit_2 import _create_vision_transformer
import torch
import time
from timm.models import build_model_with_cfg, create_model
import torch.nn as nn
from tqdm import tqdm
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
nc = 3
b_size = 16
num_epochs = 1

"""model = _create_vision_transformer(
        'vit_base_patch16_224', 
        dwa_dilation_factor=2, 
        drop_path_rate=0.1,
        num_classes=10).to(device)"""

model = create_model(
        'vit_base_patch16_224', 
        pretrained=True,
        drop_path_rate=0.1,
        num_classes=nc)
base_total_params = sum(p.numel() for p in model.parameters() )
print("INFO: Base-ViT:", base_total_params, "trainable parameters")

model = create_model(
        'vit_base_patch16_224',
        pretrained=True, 
        dwa_dilation_factor=2, 
        drop_path_rate=0.1,
        num_classes=nc).to(device) 

total_params = sum(p.numel() for p in model.parameters() )
print("INFO: DenseFormer:", total_params, "trainable parameters")
perc_inc_params = round((total_params-base_total_params)*100/base_total_params, 2)
print("INFO:", f"{perc_inc_params}% more trainable parameters than the base-ViT.")

# batch = torch.randn((1,3,224,224)).to(device)  #B, C, H, W
# print("Shape : ", batch.shape)

# start = time.time()
# output = model(batch)
# end = time.time()
# print("Duration: ", end-start)

# batch = torch.randn((5,3,224,224))
# label = torch.Tensor([0])
# print("Shape : ", batch.shape)

# start = time.time()
# output = model(batch)
# end = time.time()
# print("Duration 5: ", end-start)

# print("OUTPUT shape: ", output.shape)

optim = torch.optim.Adam(
            params=model.parameters(), lr=3e-4, weight_decay=1e-6
        )
criterion = nn.CrossEntropyLoss()
model.train()

''' create random dataloader '''
dataloader = []

for i in range(5000):
        labels = torch.tensor([random.choice([class_idx for class_idx in range(nc)]) for _ in range(b_size)])
        batch = (torch.randn(b_size, 3, 224, 224), labels)
        dataloader.append(batch)

for epoch in range(num_epochs):
    for batch in tqdm(dataloader):
        imgs = batch[0]
        labels = batch[1]
        scores = model(imgs)
        loss = criterion(scores, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()