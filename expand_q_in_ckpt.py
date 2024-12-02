import torch
import torch.nn as nn

ckpt_path = "/ckpt/ade20k_dv3_1-100_120e8/iter_80000.pth"
output_path = "/ckpt/ade20k_dv3_1-100_120e8/ckpt.pth"

pre_num_cls = 100
new_num_cls = 50

with open(ckpt_path, "rb") as f:
    ckpt = torch.load(f, map_location="cpu")
state_dict = ckpt["state_dict"]
q = state_dict["decode_head.q.weight"]
assert q.shape[0] == pre_num_cls
init = nn.Embedding(new_num_cls+q.shape[0], q.shape[1]).weight
init.requires_grad = False
# init *= 0.0
init[:q.shape[0]] = q
state_dict["decode_head.q.weight"] = init

with open(output_path, "wb") as f:
    torch.save(ckpt, f)

print()