from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
import torch

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)


from mmseg.models.segmentors.encoder_decoder import EncoderDecoder as BaseEncoderDecoder

@MODELS.register_module()
class EncoderDecoderCL_m2f(BaseEncoderDecoder):
    def __init__(self, 
                pre_num_cls: int,
                new_num_cls: int,
                temperature: float = 1.0,
                cls_factor: float = 5.0,
                attn_factor: float = 10.0,
                zero_gt: bool = True,
                freeze_backbone: bool = True,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        # assert pre_num_cls + new_num_cls == kwargs["decode_head"]["num_classes"]
        # self.backbone_freeze = copy.deepcopy(self.backbone)
        # for param in self.backbone_freeze.parameters():
        #     param.requires_grad = False
        self.zero_gt = zero_gt
        decode_head_cfg = kwargs["decode_head"]
        decode_head_cfg["num_classes"] = pre_num_cls
        # self._init_decode_head_freeze(decode_head_cfg)
        self.pre_num_cls = pre_num_cls
        self.new_num_cls = new_num_cls
        self.first_iter = True
        self.orig_decode_head_cfg = decode_head_cfg
        self.temperature = temperature
        self.cls_factor = cls_factor
        self.attn_factor = attn_factor
        self.freeze_backbone = freeze_backbone


    def _init_decode_head_freeze(self, decode_head: ConfigType) -> None:
        self.decode_head_freeze = MODELS.build(decode_head)
        for param in self.decode_head_freeze.parameters():
            param.requires_grad = False

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # calculate the distill targets
        with torch.no_grad():
            # x_freeze = self.backbone_freeze(inputs)
            if self.freeze_backbone:
                x_freeze = self.extract_feat(inputs)
                
            start_cls = 0
            orig_results = {}
            for i_head in range(self.current_task + 1):
                if not self.freeze_backbone:
                    backbone = getattr(self, 'backbone_freeze_{}'.format(i_head))
                    x_freeze = backbone(inputs)

                head = getattr(self, 'decode_head_freeze_{}'.format(i_head))
                end_cls = head.loss_cls.ignore_range
                head_results = head.forward(x_freeze, data_samples)
                cls_logits = head_results[0][-1][:,start_cls:end_cls]
                mask_logits = head_results[1][-1][:,start_cls:end_cls]
                # if 'aux_outputs' in head_results:

                # orig_aux_results = []
                cls_logits = []
                mask_logits = []
                for cls_, mask_ in zip(head_results[0], head_results[1]):
                    cls_logits.append(cls_[:,start_cls:end_cls])
                    mask_logits.append(mask_[:,start_cls:end_cls])


                if not orig_results:
                    orig_results[0] = cls_logits
                    orig_results[1] = mask_logits
                else:
                    for i, (pre_cls, pre_mask, cls_, mask_) in enumerate(zip(orig_results[0], orig_results[1], cls_logits, mask_logits)):
                        orig_results[0][i] = torch.cat((pre_cls, cls_), dim=1)
                        orig_results[1][i] = torch.cat((pre_mask, mask_), dim=1)

                start_cls = end_cls
            # orig_results = self.decode_head_freeze(x_freeze)

        if self.freeze_backbone:
            x = x_freeze
        else:
            x = self.extract_feat(inputs)

        losses = dict()
        # normal mmsegmentation _decode_head_forward_train
        loss_decode, seg_logits = self.decode_head.loss(x, data_samples, self.train_cfg, return_pred=True)

        # gt = torch.stack([data_sample.gt_sem_seg.data for data_sample in data_samples], dim=0)
        # gt_mask = F.interpolate(gt.float(), size=seg_logits[1][0].shape[-2:], mode='nearest').int()
        # gt_mask = gt_mask != self.decode_head.ignore_index

        # distill loss
        for i, (cls_, mask_) in enumerate(zip(orig_results[0], orig_results[1])):
            loss_cls_diff = F.binary_cross_entropy_with_logits\
                (seg_logits[0][i][:,:self.pre_num_cls], (cls_ * self.temperature).sigmoid(), reduction="mean")
            
            # pseudo_mask = (mask_ * self.temperature).sigmoid()
            # if self.zero_gt:
            #     pseudo_mask[gt_mask.expand_as(pseudo_mask)] = 0.0

            loss_attn_diff = F.binary_cross_entropy_with_logits\
                (seg_logits[1][i][:,:self.pre_num_cls], (mask_ * self.temperature).sigmoid(), reduction="mean")
                # (seg_logits[1][i][:,:self.pre_num_cls], pseudo_mask, reduction="mean")
            loss_decode["loss_cls_diff_aux{}".format(i)] = loss_cls_diff * self.cls_factor
            loss_decode["loss_attn_diff_aux{}".format(i)] = loss_attn_diff * self.attn_factor

        losses.update(add_prefix(loss_decode, 'decode'))

        return losses