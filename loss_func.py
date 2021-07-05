import torch
import torch.nn.functional as F
from utils import rotation_6d_to_matrix
import numpy as np
from utils import compute_metrics, smpl_reduced_to_full
import time

def compute_angle_dif(prediction, target, mask):
    """
    prediction (batch_size, seq_len, 15 * 6)
    """
    batch_size = prediction.shape[0]
    error_sum = 0
    for batch_pre, batch_target, batch_mask in zip(prediction, target, mask):
        pre_frames = []
        target_frames = []
        batch_mask = batch_mask.int().bool()

        for pre_frame, target_frame in zip(batch_pre, batch_target):
            pre_matx = rotation_6d_to_matrix(pre_frame.reshape(-1, 6)).reshape(-1)
            target_matx = rotation_6d_to_matrix(target_frame.reshape(-1, 6)).reshape(-1)
            pre_frames.append(pre_matx)
            target_frames.append(target_matx)
        pre_frams = smpl_reduced_to_full(torch.stack(pre_frames))
        target_frams = smpl_reduced_to_full(torch.stack(target_frames))
        seq_angel_error, _ = compute_metrics(pre_frams, target_frams)
        angle_error = np.mean(np.mean(seq_angel_error[batch_mask], axis=1))
        error_sum += angle_error
    return error_sum / batch_size


def poseLoss(output, target, mask):
    batch_size, seq_len, dim = output.shape
    mask = mask.int().bool()
    result = output - target
    result = torch.sum(result * result, dim=-1)
    result = result[mask] / batch_size
    return torch.sum(result)


# def poseLoss(output, target, mask):
#     batch_size, seq_len,  dim = output.shape
#     #mask = torch.unsqueeze(mask, dim=2).repeat([1, 1, output.shape[-1]]).int().bool()
#    # output = torch.masked_select(output, mask).reshape(batch_size, -1, dim)
#     #target = torch.masked_select(target, mask).reshape(batch_size, -1, dim)
#     # output = output[mask]
#     # target = target[mask]
#     # c = torch.norm(output - target, p=2, dim=-1)
#     # result = c * c
#     result = torch.zeros(1).cuda()
#     for batch_out, batch_target, batch_mask in zip(output, target, mask):
#         batch_mask = batch_mask.int().bool()
#         batch_out = batch_out[batch_mask]
#         batch_target = batch_target[batch_mask]
#         c = torch.norm(batch_out - batch_target, p=2, dim=-1)
#         result = result + (c * c).mean(0)

#     return result / batch_size

def crossEntropy(output, target, mask):
    stime = time.time()
    batch_size, seq_len, dim = output.shape

    result = torch.zeros(1).cuda()
    for batch_out, batch_target, batch_mask in zip(output, target, mask):
        batch_mask = batch_mask.int().bool()
        batch_out = batch_out[batch_mask]
        batch_target = batch_target[batch_mask]

        term = - batch_target * torch.log(batch_out) - (1 - batch_target) * torch.log(1 - batch_out)

        term = torch.sum(term, dim=-1)
        result = result + term.mean(0)
    #print("transB1",time.time() - stime)
    return result / batch_size


def ver_loss(output, target, n):
    T, dim = output.shape
    s = torch.zeros(1).cuda()
    result = output - target
    cnt = 0
    for m in range(T // n + 1):
        if m == m * n + n - 1:
            continue
        in_ = result[m * n: m * n + n - 1, :]
        if in_.shape[0] == 0:
            continue
        c = torch.norm(in_, p=2, dim=-1)
        # print(c)
        s = s + (c * c).mean(0)
        #if torch.isnan(s): print('output', output[0][0] )
        cnt += 1
    # print(s)
    return s / cnt if cnt != 0 else s


def ver_n_loss(output, target, mask):
    stime = time.time()
    batch_size, seq_len, dim = output.shape
    result = torch.zeros(1).cuda()
    for batch_out, batch_target, batch_mask in zip(output, target, mask):
        batch_mask = batch_mask.int().bool()
        batch_out = batch_out[batch_mask]
        batch_target = batch_target[batch_mask]
        result = result + ver_loss(batch_out, batch_target, 1) + ver_loss(batch_out, batch_target, 3) + ver_loss(
            batch_out, batch_target, 9) + ver_loss(batch_out, batch_target, 27)
    #print("transB2",time.time() - stime)
    return result / batch_size


def foot_accuracy(output, target, mask):
    stime = time.time()
    batch_size, seq_len, dim = output.shape
    acc_sum = 0
    for batch_out, batch_target, batch_mask in zip(output, target, mask):
        batch_mask = batch_mask.int().bool()
        batch_out = (batch_out[batch_mask] > 0.5).int()
        _seq = 1.0 * batch_out.shape[0]
        batch_target = batch_target[batch_mask].int()
        _acc = ((batch_out == batch_target).sum(dim=-1) == 2).sum() / _seq
        acc_sum += _acc
    #print("acc",time.time() - stime)
    return acc_sum / batch_size


if __name__ == "__main__":
    output = torch.randn((256, 20, 2))
    target = torch.randn((256, 20, 2))
    mask = torch.cat([torch.ones((256, 14)), torch.zeros((256, 6))], dim=-1)
    out = foot_accuracy(output, target, mask)
    print(out)
