import torch
from torchmetrics import F1Score
from torchmetrics import Accuracy
from torch import topk
from torch import nn
'''
def accuracy(output, target1, target2, device, lam):
    with torch.no_grad():
        correct1 = output.eq(target1).sum().item()
        correct2 = output.eq(target2).sum().item()
    # with torch.no_grad():
    #     acc1 = Accuracy(top_k=1).to(device)
    #     acc2 = Accuracy(top_k=1).to(device)
    #     print(output)
    #     print(target)
    #     targets, idx = topk(target, k=2, dim=1)
    #     print(targets)
    #     print(idx)
    #     batch_size, classes = target.shape[0], target.shape[1]
    #     tar1 = torch.zeros((batch_size, classes))
    #     tar2 = torch.zeros((batch_size, classes))
    #     tar1[0, idx[0]] = 1
    #     tar2[0, idx[1]] = 1
    return (lam * correct1 + (1 - lam) * correct2) / len(target) 
# acc1(output, tar1) * targets[0] + acc2(output, tar2) * (1 - targets[0])

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res
'''
def accuracy(output, target, device='cuda'):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, device, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def f1_score(output, target, device):
    with torch.no_grad():
        f1 = F1Score(num_classes=18, average='macro').to(device)
    return f1(output, target)

