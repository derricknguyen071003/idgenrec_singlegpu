def undecorated(func):
    """
    Remove decorators from a function to allow gradient computation.
    This is typically used to remove torch.no_grad() from the generate method.
    """
    while hasattr(func, '__wrapped__'):
        func = func.__wrapped__
    return func
