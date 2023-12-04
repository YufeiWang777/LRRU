from importlib import import_module


def get(args):
    # module_name = None
    assert len(args.model) != 0, 'no model is selected!'

    module_name = 'model.' + args.model
    module = import_module(module_name)

    return getattr(module, 'Model')
