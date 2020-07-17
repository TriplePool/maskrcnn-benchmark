import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function


def build_classifier_decoder(cfg, input_channels):
    return ClassifierDecoder(cfg, input_channels)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ClassifierDecoder(nn.Module):
    def __init__(self, cfg, input_channels):
        super(ClassifierDecoder, self).__init__()
        self.cfg = cfg
        self.classify_head = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Linear(input_channels, 16),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(16, cfg.MODEL.CLASSIFIER.NUM_CLASS)
        )

        self.domain_head = None

        if cfg.MODEL.CLASSIFIER.DECODING_TYPE == "TwoBranch":
            self.domain_head = nn.Sequential(
                nn.Dropout2d(0.5),
                nn.Linear(input_channels, 16),
                nn.LeakyReLU(),
                nn.Dropout2d(0.5),
                nn.Linear(16, cfg.MODEL.CLASSIFIER.NUM_DOMAIN)
            )
            self.domain_lambda = cfg.MODEL.CLASSIFIER.DOMAIN_LAMBDA
            self.domain_alpha = cfg.MODEL.CLASSIFIER.DOMAIN_ALPHA

    def forward(self, features, targets=None):
        classify_output = self.classify_head(features)
        domain_output = None
        if self.domain_head is not None:
            reverse_features = ReverseLayerF.apply(features, self.domain_alpha)
            domain_output = self.domain_head(reverse_features)

        loss = {}
        if self.training:
            if domain_output is not None:
                classify_output_list = classify_output.chunk(self.cfg.MODEL.CLASSIFIER.NUM_DOMAIN)
                domain_output_list = domain_output.chunk(self.cfg.MODEL.CLASSIFIER.NUM_DOMAIN)
                classify_target_list = targets.class_ids.chunk(self.cfg.MODEL.CLASSIFIER.NUM_DOMAIN)
                domain_target_list = targets.domain_ids.chunk(self.cfg.MODEL.CLASSIFIER.NUM_DOMAIN)
                loss['src_loss_class'] = F.cross_entropy(classify_output_list[0], classify_target_list[0])
                loss['src_loss_domain'] = F.cross_entropy(domain_output_list[0], domain_target_list[0])
                loss['tgt_loss_domain'] = 0
                for i in range(1, len(domain_output_list)):
                    loss['tgt_loss_domain'] += F.cross_entropy(domain_output_list[i], domain_target_list[i])
                loss['tgt_loss_domain'] *= self.domain_lambda

            else:
                classify_loss = F.cross_entropy(classify_output, targets.class_ids)
                loss['classify_loss'] = classify_loss

            return (classify_output, domain_output), loss
        else:
            res = post_process(classify_output, domain_output)
            return res, loss


def split_tensor_avg(t, parts_num):
    part_len = t.shape[0] // parts_num
    flag = 0
    res = []
    for i in range(parts_num):
        res.append(t[flag:flag + part_len, ...])
        flag += part_len
    return res


def post_process(classify_output, domain_output):
    if domain_output is None:
        domain_output = torch.zeros_like(classify_output)
    else:
        domain_output = F.softmax(domain_output, dim=-1)
    classify_output = F.softmax(classify_output, dim=-1)
    res_tensor = torch.cat([classify_output, domain_output], dim=-1)
    return torch.split(res_tensor, 1)
