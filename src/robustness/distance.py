import torch

def l2_loss(x, y):
    return torch.mean((x - y) ** 2, axis = 1)

def inv_pearson_corr(xs, ys):
    return 1 - torch.stack([torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1] for x, y in zip(xs, ys)])

def cosine_distance(x, y):
    return 1 - torch.sum(x * y, axis = 1) / torch.sqrt(torch.sum(x ** 2, axis = 1) * torch.sum(y ** 2, axis = 1))