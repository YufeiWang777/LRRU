import torch.optim as optim
import torch.optim.lr_scheduler as lrs

scheduler = None

class LRFactor:
    def __init__(self, decay, gamma):
        assert len(decay) == len(gamma)

        self.decay = decay
        self.gamma = gamma

    def get_factor(self, epoch):
        for (d, g) in zip(self.decay, self.gamma):
            if epoch < d:
                return g
        return self.gamma[-1]


def convert_str_to_num(val, t):
    val = val.replace('\'', '')
    val = val.replace('\"', '')

    if t == 'int':
        val = [int(v) for v in val.split(',')]
    elif t == 'float':
        val = [float(v) for v in val.split(',')]
    else:
        raise NotImplementedError

    return val

def make_optimizer_scheduler(args, target):
    # optimizer
    global scheduler
    if hasattr(target, 'param_groups'):
        # NOTE : lr for each group must be set by the network
        trainable = target.param_groups
    else:
        trainable = filter(lambda x: x.requires_grad, target.parameters())

    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'ADAMW':
        optimizer_class = optim.AdamW
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSProp
        kwargs_optimizer['eps'] = args.epsilon
    else:
        raise NotImplementedError

    optimizer = optimizer_class(trainable, **kwargs_optimizer)

   #  scheduler
    if args.scheduler == 'lambdaLR':
        # decay = convert_str_to_num(args.decay, 'int')
        # gamma = convert_str_to_num(args.gamma, 'float')
        decay = args.decay
        gamma = args.gamma

        assert len(decay) == len(gamma), 'decay and gamma must have same length'

        calculator = LRFactor(decay, gamma)
        scheduler = lrs.LambdaLR(optimizer, calculator.get_factor)
    elif args.scheduler == 'stepLR':
        scheduler = lrs.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_factor)
    elif args.scheduler == 'multistepLR':
        scheduler = lrs.MultiStepLR(optimizer, milestones=list(args.milestones), gamma=args.ml_gamma,
                                    last_epoch=args.last_epoch)

    return optimizer, scheduler
