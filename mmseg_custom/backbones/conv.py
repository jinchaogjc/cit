# Copyright (c) OpenMMLab. All rights reserved.


from mmengine.registry import MODELS
from torch import nn
from ..lora.layers import Conv2d

# MODELS.register_module('Conv1d', module=nn.Conv1d)
# MODELS.register_module('Conv2d', module=nn.Conv2d)
MODELS.register_module('Conv2d_lora', module=Conv2d)

# MODELS.register_module('Conv3d', module=nn.Conv3d)
# MODELS.register_module('Conv', module=nn.Conv2d)


