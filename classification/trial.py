from vit import _create_vision_transformer
import torch
import time
from timm.models import build_model_with_cfg


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = _create_vision_transformer(
        'vit_base_patch16_224', 
        dwa_dilation_factor=2, 
        drop_path_rate=0.1,
        num_classes=10).to(device)

# model = model.cuda()

batch = torch.randn((5,3,224,224)).to(device)
print("Shape : ", batch.shape)

start = time.time()
output = model(batch)
end = time.time()
print("Duration: ", end-start)

# batch = torch.randn((5,3,224,224))
# label = torch.Tensor([0])
# print("Shape : ", batch.shape)

# start = time.time()
# output = model(batch)
# end = time.time()
# print("Duration 5: ", end-start)

print("OUTPUT shape: ", output.shape)
