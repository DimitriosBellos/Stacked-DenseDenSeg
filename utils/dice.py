import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, smooth=0, eps=1e-8):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        if weight is not None:
            if type(weight) == torch.Tensor:
                self.weight = weight
            else:
                self.weight = torch.Tensor(weight)
        else:
            self.weight = weight
        self.smooth = smooth
        self.eps = eps

    def forward(self, predictions, target):
        predictions = F.softmax(predictions)
        if target.is_cuda:
            y = torch.cuda.FloatTensor(predictions.shape).zero_()
        else:
            y = torch.FloatTensor(predictions.shape).zero_()
        target = target.unsqueeze(1)
        y.scatter_(1, target, 1)
        intersection = predictions * y
        while intersection.dim() > 2:
            intersection = torch.sum(intersection, 2)
            y = torch.sum(y, 2)
            predictions = torch.sum(predictions, 2)
        intersection = torch.sum(intersection, 0)
        y = torch.sum(y, 0)
        predictions = torch.sum(predictions, 0)
        if self.weight is None:
            num = torch.sum(2 * intersection + self.smooth)
        else:
            num = torch.sum(2 * intersection * self.weight + self.smooth)
        den = torch.sum(y + predictions + self.smooth + self.eps)
        return 1 - num / den


if __name__ == '__main__':
    y = torch.LongTensor(16, 64, 64, 64).random_(11)
    yp = torch.FloatTensor(16, 11, 64, 64, 64).uniform_(-10, 10)
    criterion = DiceLoss(11)
    loss = criterion(yp, y)
    print('OK')
