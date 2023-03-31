# Custom Contrastive Loss
import torch.nn
import torch.nn.functional as F
from torch import tensor


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        # print((1 - label.float()) * torch.pow(euclidean_distance, 2))
        # print(torch.clamp(self.margin - euclidean_distance, min=0.0))
        loss_contrastive = torch.mean((1 - label.float()) * torch.pow(euclidean_distance, 2) +
                                      label.float() * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive / 10

