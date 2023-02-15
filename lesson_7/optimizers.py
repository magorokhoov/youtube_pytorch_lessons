import torch
import torch.nn as nn
import torch.nn.functional as F

def get_optimizer(params, option_optimizer:dict):
    name = option_optimizer['name']

    if name == 'sgd':
        lr = float(option_optimizer['lr'])
        momentum = option_optimizer.get('momentum', 0.9)
        dampening = option_optimizer.get('dampening', 0.0)
        nesterov = option_optimizer.get('nesterov', False)
        weight_decay = option_optimizer.get('weight_decay', 0.0)

        optimizer = torch.optim.SGD(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            weight_decay=weight_decay
        )

    elif name in ('adam', 'adamw'):
        lr = float(option_optimizer['lr'])
        beta1 = option_optimizer.get('beta1', 0.9)
        beta2 = option_optimizer.get('beta2', 0.999)
        eps = option_optimizer.get('eps', 1e-8)
        weight_decay = option_optimizer.get('weight_decay', 0.0)

        if name == 'adam':
            optimizer = torch.optim.Adam(
                params=params,
                lr=lr,
                betas=(beta1,beta2),
                eps=eps,
                weight_decay=weight_decay
            )
        elif name == 'adamw':
            optimizer = torch.optim.AdamW(
                params=params,
                lr=lr,
                betas=(beta1,beta2),
                eps=eps,
                weight_decay=weight_decay
            )

    else:
        raise NotImplementedError(f'optimizer [{name}] is not implemented')
    
    return optimizer