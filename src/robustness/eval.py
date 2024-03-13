from typing import Callable

import torch, torchvision

from .distance import cosine_distance

class Robustness(object):
    
    def __init__(self,
                 distance_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cosine_distance,
                 margin : float = 0.5):
        
        self.distance_fn = distance_fn
        self.margin = margin
    
    def __call__(self, preds0 : torch.Tensor, preds1 : torch.Tensor):
        n_preds = len(preds0)
        
        qs = preds0.flatten(1)
        k1 = preds1.flatten(1)
        
        k0 = torch.stack([k1[i-1] for i in range(n_preds, 0, -1)]) # reverse second array to get negatives
        k0 = torch.stack([k0[-2], k0[-1], *k0[:-2]]) # shift array by 2 so there is no match in the middle
        
        neg_distance = self.distance_fn(qs, k0)
        pos_distance = self.distance_fn(qs, k1)
        
        return torch.maximum(torch.tensor(0), pos_distance - neg_distance + self.margin)

@torch.no_grad()
def predict_w_model(model : torch.nn.Module, imgs : torch.Tensor,
                    batch_size : int = 32, device : torch.device = 'cuda:0',
                    level : int = -2, pool : bool = True, *args, **kwargs):
    
    model = model.to(device)
    batch_preds = []
    
    dataloader = torch.utils.data.DataLoader(imgs, batch_size, shuffle = False)
    for i, x in enumerate(dataloader):
        x = x.to(device)
        y_hat, inner_reprs = model(x, return_skip_vals = True)
        batch_preds.append((y_hat.cpu(), [inner_repr.cpu() for inner_repr in inner_reprs]))
        del x, y_hat, inner_reprs
    preds = [[] for _ in range(1 + len(batch_preds[0][1]))]
    for batch_pred in batch_preds:
        for i, inner_repr in enumerate(batch_pred[1]):
            preds[i].append(inner_repr)
        preds[i+1].append(batch_pred[0])
    
    pred = preds[level]
    if pool:
        return torch.mean(pred.flatten(2), dim = 2)
    else:
        return pred

def eval_encoder(model : torch.nn.Module, imgs : torch.Tensor, scorer : Robustness,
                 level : int, pool: bool, *args, **kwargs):
    encoder = getattr(model, 'model', model).layers[0]
    encoder.eval()
    
    augmentation = torchvision.transforms.ColorJitter(brightness = 0.1,
                                                      contrast = 0.05,
                                                      hue = 0.05,
                                                      saturation = 0.1)
    
    imgs0, imgs1 = augmentation(imgs), augmentation(imgs)
    preds0 = predict_w_model(encoder, imgs0, level = level, pool = pool, *args, **kwargs)
    preds1 = predict_w_model(encoder, imgs1, level = level, pool = pool, *args, **kwargs)
    
    return scorer(preds0, preds1)