import torch
from torch import nn
from torch.nn import functional as F


def build_classifier_decoder(cfg, input_channels):
    return ClassifierDecoder(cfg, input_channels)


class ClassifierDecoder(nn.Module):
    def __init__(self, cfg, input_channels):
        super(ClassifierDecoder, self).__init__()
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

    def forward(self, features, targets=None):
        classify_output = self.classify_head(features)
        domain_output = None
        if self.domain_head is not None:
            domain_output = self.domain_head(features)

        loss = {}
        if self.training:
            classify_loss = F.cross_entropy(classify_output, targets.class_ids)
            loss['classify_loss'] = classify_loss
            if domain_output is not None:
                domain_loss = F.cross_entropy(domain_output, targets.domain_ids)
                loss['domain_loss'] = domain_loss

        return (classify_output, domain_output), loss
