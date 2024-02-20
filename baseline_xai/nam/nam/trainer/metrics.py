def mae(logits, targets):
    return ((logits.view(-1) - targets.view(-1)).abs().sum() / logits.numel()).item()

def mse(logits, targets):
    return (((logits.view(-1) - targets.view(-1))**2).sum() / logits.numel()).item()


def accuracy(logits, targets):
    return (((targets.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() / targets.numel()).item()


# def cal_f1_score(logits, targets): 
#     logits = logits.view(-1)
#     targets = targets.view(-1)
#     print('logits: ', logits)
#     print("targets: ", targets)
