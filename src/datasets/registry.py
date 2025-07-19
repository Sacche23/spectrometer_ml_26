DATASET_REGISTRY = {}

def register_dataset(name):
    '''
    Call "@register_dataset(name)" above
    class declaration to register name as a valid
    dataset parameter.
    '''
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def get_dataset(name):
    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
