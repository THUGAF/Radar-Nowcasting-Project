import torch
import numpy as np


def _count(pred, truth, threshold):
    pred = pred.cpu().transpose(2, 0).squeeze(0)
    truth = truth.cpu().transpose(2, 0).squeeze(0)
    
    batch_size, _, _, _ = pred.size()
    hits = []
    misses = []
    false_alarms = []
    correct_rejections = []

    for b in range(batch_size):
        stat = 2 * (truth[b] > threshold).int() + (pred[b] > threshold).int()
        hits.append(torch.sum(stat == 3).item())
        misses.append(torch.sum(stat == 2).item())
        false_alarms.append(torch.sum(stat == 1).item())
        correct_rejections.append(torch.sum(stat == 0).item())
    
    hits, misses, false_alarms, correct_rejections = np.array(hits), np.array(misses), np.array(false_alarms), np.array(correct_rejections)
    return hits, misses, false_alarms, correct_rejections

def _count_each(pred, truth, threshold):
    pred = pred.cpu().transpose(2, 0).squeeze(0)
    truth = truth.cpu().transpose(2, 0).squeeze(0)

    batch_size, seq_len, _, _ = pred.size()
    hits = np.zeros((batch_size, seq_len))
    misses = np.zeros((batch_size, seq_len))
    false_alarms = np.zeros((batch_size, seq_len))
    correct_rejections = np.zeros((batch_size, seq_len))

    for b in range(batch_size):
        for s in range(seq_len):
            stat = 2 * (truth[b, s] > threshold).int() + (pred[b, s] > threshold).int()
            hits[b, s] = torch.sum(stat == 3).item()
            misses[b, s] = torch.sum(stat == 2).item()
            false_alarms[b, s] = torch.sum(stat == 1).item()
            correct_rejections[b, s] = torch.sum(stat == 0).item()

    return hits, misses, false_alarms, correct_rejections

def evaluate(pred, truth, threshold=0.2):
    r"""To calculate the mean value of POD, FAR, CSI and HSS for the prediction.
        Arguments:
            pred (pytorch tensor): The prediction sequence in tensor form with 5D shape `(S, B, C, H, W)`
            truth (pytorch tensor): The ground truth sequence in tensor form with 5D shape `(S, B, C, H, W)`
            threshold (float, optional): The threshold of POD, FAR, CSI and HSS. Range: (0, 1). Default: 0.2
        Return:
            (list): A list of POD, FAR, CSI and HSS
    """
    h, m, f, c = _count(pred, truth, threshold)
    eps = 1e-4
    pod = np.mean(h / (h + m + eps))
    far = np.mean(f / (h + f + eps))
    csi = np.mean(h / (h + m + f + eps))
    hss = np.mean(2 * (h * c - m * f) / ((h + m) * (m + c) + (h + f) * (f + c) + eps))
    return [pod, far, csi, hss]

def evaluate_each(pred, truth, threshold=0.2):
    r"""To calculate POD, FAR, CSI and HSS for the prediction at each time step.
        Arguments:
            pred (pytorch tensor): The prediction sequence in tensor form with 5D shape `(S, B, C, H, W)`
            truth (pytorch tensor): The ground truth sequence in tensor form with 5D shape `(S, B, C, H, W)`
            threshold (float, optional): The threshold of POD, FAR, CSI and HSS. Range: (0, 1). Default: 0.2
        Return:
            (ndarray): A 2D array of POD, FAR, CSI and HSS at each time step
    """
    h, m, f, c = _count_each(pred, truth, threshold)
    eps = 1e-4
    pod = np.mean(h / (h + m + eps), axis=0)
    far = np.mean(f / (h + f + eps), axis=0)
    csi = np.mean(h / (h + m + f + eps), axis=0)
    hss = np.mean(2 * (h * c - m * f) / ((h + m) * (m + c) + (h + f) * (f + c) + eps), axis=0)
    return [pod, far, csi, hss]
