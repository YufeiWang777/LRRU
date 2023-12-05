from importlib import import_module
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def get(args):
    summary_name = args.summary_name
    module_name = 'summary.' + summary_name.lower()
    module = import_module(module_name)

    return getattr(module, 'Summary')


class BaseSummary(SummaryWriter):
    def __init__(self, log_dir, mode, args):
        super(BaseSummary, self).__init__(log_dir=log_dir + '/' + mode)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.f_loss = '{}/loss_{}.txt'.format(log_dir, mode)
        self.f_metric = '{}/metric_{}.txt'.format(log_dir, mode)

        f_tmp = open(self.f_loss, 'w')
        f_tmp.close()
        f_tmp = open(self.f_metric, 'w')
        f_tmp.close()

    def add(self, loss=None, metric=None, log_itr=None):
        # loss and metric should be numpy arrays
        if loss is not None:
            self.loss.append(loss.data.cpu().numpy())
        if metric is not None:
            self.metric.append(metric.data.cpu().numpy())

    def update(self, global_step, sample, output):
        self.loss = np.concatenate(self.loss, axis=0)
        self.metric = np.concatenate(self.metric, axis=0)

        self.loss = np.mean(self.loss, axis=0)
        self.metric = np.mean(self.metric, axis=0)

        self.reset()
        pass

    def make_dir(self, epoch, idx):
        pass

    def save(self, epoch, idx, sample, output):
        pass



