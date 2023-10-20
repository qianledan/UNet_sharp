import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
#         self.CE_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
#         logpt = self.CE_loss(output, target)
        logpt = F.binary_cross_entropy_with_logits(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()
