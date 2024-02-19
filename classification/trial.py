from vit import *
import torch
import time

model = vit_base_patch8_224(dwa_dilation_factor=1, drop_path_rate=0.1)

batch = torch.randn((1,3,224,224))
label = torch.Tensor([0])
print("Shape : ", batch.shape)

start = time.time()
output = model(batch)
end = time.time()
print("Duration: ", end-start)


batch = torch.randn((5,3,224,224))
label = torch.Tensor([0])
print("Shape : ", batch.shape)

start = time.time()
output = model(batch)
end = time.time()
print("Duration 5: ", end-start)

print("OUTPUT shape: ", output.shape)
