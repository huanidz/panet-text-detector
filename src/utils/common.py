def count_parameters(model, count_non_trainable=False):
    if count_non_trainable:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)