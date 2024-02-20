from vit import VisionTransformer
import torch
import time
from timm.models import build_model_with_cfg


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = build_model_with_cfg(VisionTransformer,
        'vit_base_patch16_224', 
        pretrained=False,
        dwa_dilation_factor=1, 
        drop_path_rate=0.1,
        num_classes=10)

model = model.cuda()

batch = torch.randn((1,3,224,224)).cuda()
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
