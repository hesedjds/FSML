import torch
import numpy as np

from sklearn.metrics import auc


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def recall(embeddings, labels, K=[]):
    """
    Multiply by 10 for numerical stability.
    It does not affect recall.
    """
    D = pdist(embeddings * 10, squared=True)

    knn_inds = D.topk(1 + max(K), dim=1, largest=False, sorted=True)[1][:, 1:]
    """
    Check if, knn_inds contain index of query image.
    """
    assert ((knn_inds == torch.arange(0, len(labels), device=knn_inds.device).unsqueeze(1)).sum().item() == 0)

    selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
    correct_labels = labels.unsqueeze(1) == selected_labels

    recall_k = []
    for k in K:
        correct_k = (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)
    return recall_k


def mAP(embeddings, labels):
    """
    Multiply by 10 for numerical stability.
    It does not affect average precision. 
    """

    def ap(precision, recall):
        mrec = []
        mrec.append(0)
        [mrec.append(e.item()) for e in recall[:, k]]
        mpre = []
        mpre.append(0)
        [mpre.append(e.item()) for e in precision[:, k]]
        mrec.append(1.)
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        li = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                li.append(i + 1)
        ap = 0
        for i in li:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap

    D = pdist(embeddings * 10, squared=True)

    labels_unique = labels.unique()
    labels = (labels.unsqueeze(1) == labels_unique).nonzero()[:, 1].contiguous()

    sorted_D, sorted_indices = torch.sort(D, dim=0)
    labels_sorted = labels[sorted_indices[1:]]
    correct = labels == labels_sorted

    labels_num = (labels == torch.arange(labels.max()+1, device=embeddings.device).view(-1, 1)).sum(1)
    labels_num_each = labels_num[labels] - 1

    correct_cumsum = torch.cumsum(correct, dim=0).float()
    precision = correct_cumsum / (torch.arange(correct_cumsum.size(0), device=embeddings.device).view(-1, 1) + 1).float()
    recall = correct_cumsum / labels_num_each.float()
    
    precision = precision.cpu()
    recall = recall.cpu()
    mAP = 0. 
    for k in range(embeddings.size(0)): 
        mAP += auc(recall[:, k], precision[:, k])
    mAP /= embeddings.size(0)

    return mAP