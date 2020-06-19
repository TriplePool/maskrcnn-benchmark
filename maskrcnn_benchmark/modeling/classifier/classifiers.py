import torch
from torch import nn

from maskrcnn_benchmark.modeling.classifier.decoder.decoder import build_classifier_decoder
from maskrcnn_benchmark.modeling.classifier.encoder.encoder import build_classifier_encoder
from maskrcnn_benchmark.structures.image_list import to_image_list


class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.encoder = build_classifier_encoder(cfg)
        self.decoder = build_classifier_decoder(cfg)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.encoder(images.tensors)

        result, decoder_losses = self.decoder(features, targets)

        if self.training:
            return decoder_losses

        return result


def build_classifier_model(cfg):
    return Classifier(cfg)
