import torch
import torch.nn as nn
import torch.nn.functional as F
from .metric_model import MetricModel
from core.utils import accuracy
import numpy as np


class FrequencyMaskGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, features, labels):
        # DCT transform
        dct = torch.fft.rfft2(x)

        # Generate Grad-CAM attention maps using GAP instead of gradients
        batch_size = features.size(0)
        channel_weights = self.gap(features)  # Use GAP directly instead of gradients

        attention = torch.zeros_like(features)
        for i in range(batch_size):
            # Weight each channel and sum
            weighted_features = features[i] * channel_weights[i]
            attention[i] = F.relu(weighted_features.sum(dim=0, keepdim=True))

        # Normalize attention
        attention = attention / (attention.max() + 1e-8)

        # Upsample attention maps
        attention = F.interpolate(attention, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # Generate frequency mask
        mask = 1 - torch.sigmoid(attention)

        # Expand mask to match DCT dimensions
        if len(mask.shape) < len(dct.shape):
            mask = mask.unsqueeze(-1).expand_as(dct)

        # Apply mask in frequency domain
        masked_dct = dct * mask
        unmasked_dct = dct * (1 - mask)

        # Inverse DCT
        masked_img = torch.fft.irfft2(masked_dct, s=x.shape[-2:])
        unmasked_img = torch.fft.irfft2(unmasked_dct, s=x.shape[-2:])

        return masked_img, unmasked_img, mask

class MultiLevelMetrics(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.triplet_loss = nn.TripletMarginLoss(margin=0.3)

    def forward(self, support_feat, query_feat, support_y, query_y,
                masked_support, unmasked_support, masked_query, unmasked_query):
        losses = {}

        # Sample-wise triplet loss
        triplet_loss = self.triplet_loss(query_feat,
                                         unmasked_query,
                                         masked_query)
        losses['triplet'] = triplet_loss

        # Class-wise contrastive loss
        support_proto = self._get_prototypes(support_feat, support_y)
        masked_support_proto = self._get_prototypes(masked_support, support_y)

        logits = F.cosine_similarity(query_feat.unsqueeze(1),
                                     support_proto.unsqueeze(0), dim=-1) / self.temperature

        contrastive_loss = F.cross_entropy(logits, query_y)
        losses['contrastive'] = contrastive_loss

        # Augmented classification loss
        aug_support = torch.cat([support_feat, unmasked_support], dim=0)
        aug_support_y = torch.cat([support_y, support_y], dim=0)
        aug_proto = self._get_prototypes(aug_support, aug_support_y)

        aug_logits = F.cosine_similarity(query_feat.unsqueeze(1),
                                         aug_proto.unsqueeze(0), dim=-1) / self.temperature
        aug_loss = F.cross_entropy(aug_logits, query_y)
        losses['augmented'] = aug_loss

        return losses

    def _get_prototypes(self, features, targets):
        classes = torch.unique(targets)
        protos = []
        for c in classes:
            protos.append(features[targets == c].mean(0))
        return torch.stack(protos)


class FGFL(MetricModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask = FrequencyMaskGenerator()
        self.metrics = MultiLevelMetrics()
        self.loss_weights = {'triplet': 1.0, 'contrastive': 1.0, 'augmented': 1.0}

    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))

        # Extract features
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)

        # Generate frequency masks and masked images
        masked_support, unmasked_support, _ = self.freq_mask(support_feat, feat, support_target)
        masked_query, unmasked_query, _ = self.freq_mask(query_feat, feat, query_target)

        # Get prototypes
        support_proto = self._get_prototypes(support_feat, support_target)

        # Calculate cosine similarities
        logits = F.cosine_similarity(query_feat.unsqueeze(1),
                                     support_proto.unsqueeze(0), dim=-1)

        acc = accuracy(logits, query_target)
        return logits, acc

    def set_forward_loss(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        # Extract features
        with torch.no_grad():  # 提取特征时不需要梯度
            feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)

        # 确保特征图维度正确
        if len(support_feat.shape) == 2:
            support_feat = support_feat.view(*support_feat.shape, 1, 1)
        if len(query_feat.shape) == 2:
            query_feat = query_feat.view(*query_feat.shape, 1, 1)

        # Generate frequency masks and masked images
        try:
            masked_support, unmasked_support, _ = self.freq_mask(support_feat, support_feat, support_target)
            masked_query, unmasked_query, _ = self.freq_mask(query_feat, query_feat, query_target)
        except RuntimeError as e:
            print(f"Error in mask generation: {e}")
            print(f"Support feat shape: {support_feat.shape}")
            print(f"Query feat shape: {query_feat.shape}")
            raise

        # Calculate multi-level metrics
        losses = self.metrics(support_feat, query_feat, support_target, query_target,
                              masked_support, unmasked_support, masked_query, unmasked_query)

        # Weighted sum of losses
        total_loss = sum(self.loss_weights[k] * v for k, v in losses.items())

        # Get predictions for accuracy
        support_proto = self._get_prototypes(support_feat, support_target)
        logits = F.cosine_similarity(query_feat.unsqueeze(1),
                                     support_proto.unsqueeze(0), dim=-1)
        acc = accuracy(logits, query_target)

        return logits, acc, total_loss

    def _get_prototypes(self, features, targets):
        classes = torch.unique(targets)
        protos = []
        for c in classes:
            protos.append(features[targets == c].mean(0))
        return torch.stack(protos)