import torch
from train_tools.models import MEDIARMamba, MEDIARFormer
from thop import profile
from thop import clever_format

model = MEDIARFormer().cuda().eval()
data = torch.rand((1,3,1024,1024)).cuda()

MACs, params = profile(model, inputs=[data])  
MACs, params = clever_format([MACs, params], "%.3f") 

print("MACs: ", MACs)
print("params: ", params)
