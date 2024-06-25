from torch import nn
from transformers import AutoModel


class CustomDinoV2(nn.Module):
    def __init__(self, n_cls):
        super(CustomDinoV2, self).__init__()
        self.base_model = AutoModel.from_pretrained('facebook/dinov2-base')
        for param in self.base_model.parameters():
            param.requires_grad = False

        num_features = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, n_cls),
        )

    def forward(self, inputs):
        x = self.base_model(inputs)
        x = self.classifier(x.pooler_output)
        return x