# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy with label smoothing.

    smoothing=0 → standard cross entropy
    smoothing>0 → distributes some probability mass to other classes
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        """
        logits: (batch_size, num_classes)
        target: (batch_size,)
        """

        num_classes = logits.size(1)

        # Convert targets to one-hot
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=1)

        loss = - (true_dist * log_probs).sum(dim=1)

        return loss.mean()
