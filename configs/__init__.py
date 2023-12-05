from importlib import import_module

def get(arg=None):

    config_name = 'get_cfg_defaults'
    module_name = 'configs.config'
    module = import_module(module_name)
    get_config = getattr(module, config_name)
    cfg = get_config()

    if arg is not None:
        cfg.defrost()
        cfg.merge_from_file('configs/' + arg.configuration)
        cfg.num_gpus = len(cfg.gpus)
        cfg.project_name = arg.project_name
        cfg.freeze()
        args_config = cfg
    else:
        args_config = cfg

    return args_config


