import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def hard_region_Loss(predictions, labels, boundary_mask):
       
# 找到边界左右两边的像素
   
        # 找到边界左右两边的标签
        left_boundary_labels = labels*boundary_mask
        right_boundary_labels = labels*boundary_mask
        right_boundary_labels=right_boundary_labels[:,1:,:,:]
        #torch.Size([2, 112, 112, 80])
#########################
# 根据边界掩码和过渡区域掩码，提取边界像素和过渡区域像素
        # boundary_pixels = predictions*boundary_mask
        # transition_pixels = predictions*boundary_mask
        boundary_pixels = predictions
        transition_pixels = predictions
        # 根据掩码提取相应的标签
        boundary_labels = labels*boundary_mask
        transition_labels = labels*boundary_mask
        print('label',labels.shape)
        # 使用局部卷积来捕获过渡信息
        trans_pixels = local_conv(transition_pixels, 3)
        # 计算边界损失，考虑左右两边像素
        #[:labeled_bs, 1, :, :, :], v_label[:labeled_bs] )
        boundary_dist = softmax_mse_loss(boundary_pixels[:, 1, :, :, :], labels)
        # 计算过渡区域损失
        transition_dist = softmax_mse_loss(trans_pixels[:, 1, :, :, :], labels)
        #ssssprint('transition_dist',transition_dist)
        boundary_loss = torch.sum(boundary_mask * boundary_dist) / (torch.sum(boundary_mask) + 1e-16)
        print('boundary_loss',boundary_loss)
       # transition_loss = torch.sum(boundary_mask * transition_dist) / (torch.sum(boundary_mask) + 1e-16)
        #print('transition_loss',transition_loss)
        # 组合总损失
        total_loss = 0.5* boundary_loss #+ 0.5 * transition_loss# + left_boundary_loss #+ right_boundary_loss
    
        return total_loss
def local_conv(x, kernel_size=3):
        padding = (kernel_size - 1) // 2
        local_convolution = nn.Conv3d(in_channels=2, out_channels=2, kernel_size=kernel_size, padding=padding, bias=False).to(device)
        local_convolution.weight.data.fill_(1.0).to(device) # 权重全为1，即平均卷积
        local_convolution.weight.requires_grad = False  # 权重不需要训练
        local_features = local_convolution(x).to(device)  
        return local_features    
def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
