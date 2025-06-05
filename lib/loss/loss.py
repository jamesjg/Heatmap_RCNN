
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Callable, Optional

class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss. paper ref: 'Adaptive Wing Loss for Robust Face
    Alignment via Heatmap Regression' Wang et al. ICCV'2019.
    Args:
        alpha (float), omega (float), epsilon (float), theta (float)
            are hyper-parameters.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 alpha=2.1,
                 omega=14,
                 epsilon=1,
                 theta=0.5,
                 use_target_weight=False,
                 loss_weight=1.):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def criterion(self, pred, target):
        """Criterion of wingloss.
        Note:
            batch_size: N
            num_keypoints: K
        Args:
            pred (torch.Tensor[NxKxHxW]): Predicted heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
        """
        H, W = pred.shape[2:4]
        delta = (target - pred).abs()

        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        ) * (self.alpha - target) * (torch.pow(
            self.theta / self.epsilon,
            self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        losses = torch.where(
            delta < self.theta,
            self.omega *
            torch.log(1 +
                      torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C)

        return torch.mean(losses)

    def forward(self, output, target, target_weight=1):
        """Forward function.
        Note:
            batch_size: N
            num_keypoints: K
        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            loss = self.criterion(output * target_weight.unsqueeze(-1),
                                  target * target_weight.unsqueeze(-1))
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


class SoftKLLoss(nn.KLDivLoss):
    """
        l_n = (Softmax y_n) \cdot \left( \log (Softmax y_n) - \log (Softmax x_n) \right)
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False) -> None:
        super(SoftKLLoss,self).__init__(size_average=size_average, reduce=reduce, reduction=reduction, log_target=log_target)
        self.logSoftmax = nn.LogSoftmax(dim=-1)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """  
        input is pred
        target is gt
        """
        # input = input.view(-1, input.shape[-2]*input.shape[-1])
        input = input.reshape(-1,input.shape[-2]*input.shape[-1])
        # target = target.view(-1, target.shape[-2]*target.shape[-1])
        target = target.reshape(-1,target.shape[-2]*target.shape[-1])
        input = self.logSoftmax(input)
        # target = self.Softmax(target) # not sure should softmax for target
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)

class CE_Loss(nn.CrossEntropyLoss):
    def __init__(self, class_nums:int, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean') -> None:
        super(CE_Loss, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.class_nums = class_nums

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(input.reshape(-1, self.class_nums), target.reshape(-1, self.class_nums), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

class MixLoss_AWINGANDCEL(nn.CrossEntropyLoss):
    def __init__(self, alpha, class_nums:int, weight=None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MixLoss_AWINGANDCEL,self).__init__(weight=weight,size_average=size_average,reduce=reduce,reduction=reduction)
        self.alpha = alpha
        self.reduction = reduction
        self.class_nums = class_nums
        self.awing = AdaptiveWingLoss()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cls_out = F.cross_entropy(input.reshape(-1, self.class_nums), target.reshape(-1, self.class_nums),reduction=self.reduction)
        awing_out = self.awing(input, target)
        return awing_out + self.alpha * cls_out

class MixLoss_L2ANDCEL(nn.CrossEntropyLoss):

    def __init__(self, alpha, class_nums:int, weight=None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MixLoss_L2ANDCEL,self).__init__(weight=weight,size_average=size_average,reduce=reduce,reduction=reduction)
        self.alpha = alpha
        self.reduction = reduction
        self.class_nums = class_nums

    def forward(self,input:torch.Tensor, target:torch.Tensor):
        mse_out = F.mse_loss(input, target, reduction=self.reduction)
        cls_out = F.cross_entropy(input.reshape(-1, self.class_nums), target.reshape(-1, self.class_nums),reduction=self.reduction)
        return mse_out + self.alpha * cls_out


class MixLoss_L2ANDKL(SoftKLLoss):
    def __init__(self, alpha, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False) -> None:
        super(MixLoss_L2ANDKL,self).__init__(size_average=size_average,reduce=reduce,reduction=reduction,log_target=log_target)
        self.alpha = alpha
        self.reduction = reduction
        self.log_target = log_target

    def forward(self,input:torch.Tensor, target:torch.Tensor):
        mse_out = F.mse_loss(input, target, reduction=self.reduction)
        cls_out = super(MixLoss_L2ANDKL,self).forward(input, target)
        return mse_out + self.alpha * cls_out

