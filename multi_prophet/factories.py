from .prophet import Prophet

def model_pool_factory(columns=[], args_dict=None, **kwargs):
    if args_dict:
        return _different_models_pool_factory(args_dict)
    else:
        return _equal_models_pool_factory(columns, **kwargs)

def _different_models_pool_factory(args_dict):
    return {c: Prophet(**kwargs) for c, kwargs in args_dict.items()}

def _equal_models_pool_factory(columns, **kwargs):
    return {c: Prophet(**kwargs) for c in columns}
