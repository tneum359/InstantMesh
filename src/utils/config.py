import importlib

def get_obj_from_str(string, reload=False):
    """
    Gets an object (class or function) from a string specifying its path.
    Example: 'my_package.my_module.MyClass'
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # Ensure the module is imported relative to the package if necessary
    # The 'package=None' argument might need adjustment depending on context,
    # but usually None works for absolute paths.
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    """
    Instantiates an object from an OmegaConf configuration object or dict.
    Expects the config to have a 'target' key specifying the object's path
    and an optional 'params' key with instantiation parameters.
    """
    if config is None:
        return None
    if not isinstance(config, dict) and not hasattr(config, "target"):
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError(f"Expected config to be OmegaConf object or dict with key `target`, but got {type(config)}")

    if "target" not in config:
        # Allow config to be a placeholder string like '__is_first_stage__'
        # This case should have been caught by the check above if config is not a dict.
        # If config *is* a dict but missing 'target', it's an error unless it's a special internal marker.
        if config.get('__is_first_stage__', False): return None # Example of an internal marker
        if config.get('__is_unconditional__', False): return None # Example of an internal marker
        raise KeyError("Expected key `target` to instantiate.")
    
    target_str = config["target"]
    params = config.get("params", dict())
    
    if params is None: # Ensure params is a dict for ** unpacking
        params = dict()

    try:
        return get_obj_from_str(target_str)(**params)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not instantiate object from config:")
        print(f"  Target: {target_str}")
        print(f"  Params: {params}")
        print(f"  Error: {e}")
        raise e

if __name__ == '__main__':
    # Example Usage (requires OmegaConf to be installed: pip install omegaconf)
    from omegaconf import OmegaConf

    # Example 1: Basic class instantiation
    conf1 = OmegaConf.create({
        'target': 'collections.Counter', 
        'params': {'iterable': 'abracadabra'}
    })
    counter_instance = instantiate_from_config(conf1)
    print(f"Example 1 Counter: {counter_instance}")

    # Example 2: Class with no params
    conf2 = OmegaConf.create({
        'target': 'list'
        # 'params' is optional, defaults to empty dict
    })
    list_instance = instantiate_from_config(conf2)
    print(f"Example 2 List: {list_instance}")

    # Example 3: Nested instantiation (if your classes/configs support it)
    # Define a dummy class for demonstration
    class MyModule:
        def __init__(self, sub_module=None, value=0):
            self.sub_module = sub_module
            self.value = value
        def __repr__(self):
            return f"MyModule(sub_module={self.sub_module}, value={self.value})"

    conf3 = OmegaConf.create({
        'target': '__main__.MyModule', # Use __main__ because it's defined in this script for the example
        'params': {
            'value': 10,
            'sub_module': { # Nested config for another object
                 'target': 'collections.deque',
                 'params': {'iterable': [1, 2, 3]}
            }
        }
    })
    # Note: Nested instantiation requires the outer class's __init__ 
    # to handle the sub-config dict or expect an instantiated object.
    # This basic instantiate_from_config doesn't automatically instantiate nested dicts.
    # You would typically handle nesting within your class's __init__ or a custom factory.
    # For demonstration, let's assume MyModule's __init__ could handle a dict:
    # Or modify instantiate_from_config to recursively instantiate dicts with 'target'.
    
    # Basic usage will pass the dict:
    # module_instance = instantiate_from_config(conf3) 
    # print(f"Example 3 MyModule (sub_module as dict): {module_instance}") 
    
    # If you want recursive instantiation, you'd modify instantiate_from_config
    # to check if values in params are dicts with 'target' and call itself.

    print("\\nNote: Nested instantiation requires specific handling not shown in basic example.") 