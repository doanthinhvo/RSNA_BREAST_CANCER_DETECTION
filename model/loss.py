import torch
def cancer_loss(pred, target, pos_weight=None):
    return torch.nn.functional.binary_cross_entropy_with_logits(pred, target, pos_weight=torch.tensor(1.0))