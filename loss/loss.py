from . import BaseLoss
import torch


class Loss(BaseLoss):
    def __init__(self, args, loss):
        super(Loss, self).__init__(args, loss)

        self.loss_name = []

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, output, sample):
        loss_val = []
        # b, c, h, w = list(sample.size())

        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue

            # pred = output['pred']
            # gt = sample['gt']
            pred, gt = output, sample

            if loss_type in ['L1', 'L2', 'Ls', 'L3']:
                loss_tmp = loss_func(pred, gt)
                # tmp = b * c * h * w * 0.16
                # loss_tmp = loss_tmp.sum() / (b * c * h * w * 0.16)
            else:
                raise NotImplementedError

            loss_tmp = loss['weight'] * loss_tmp
            loss_val.append(loss_tmp)

        loss_val = torch.stack(loss_val)

        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val
