import torch.nn as nn
import torch.nn.functional as F
import torch


class BinaryCrossEntropyLoss2D(nn.Module):
    def __init__(self, weight=None, size_average=True):
        """
        Binary cross entropy loss 2D
        Args:
            weight:
            size_average:
        """
        super(BinaryCrossEntropyLoss2D, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)  # Flatten
        targets_flat = targets.view(-1)  # Flatten
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1) # [8, 388*388]
        m2 = targets.view(num, -1) # [8, 388*388]
        intersection = (m1 * m2) # 取交集

        # 即每个image都有一个score，为dice系数(一种集合相似性度量，这里其实就是准确率)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth) # [8] 
        score = 1 - score.sum() / num
        return score


class customLoss(nn.Module):
    def __init__(self, ):
        super(customLoss, self).__init__()
        self.bceloss = BinaryCrossEntropyLoss2D()
        self.sdloss = SoftDiceLoss()
    
    def forward(self, logits, labels):
        loss = self.bceloss(logits, labels) + self.sdloss(logits, labels)
        return loss



# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(pred, target):  # 这个跟上面有点不一样，上面是在dim=1上sum，即在一个图片内所有像素点，这里是对一个batch所有图片的像素点sum。
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth) # 该batch的像素级别准确率
