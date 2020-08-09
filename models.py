import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embeding = nn.Embedding(config.n_vocab,
                                     config.embed_size,
                                     padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed_size,
                            config.hidden_size,
                            config.num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed_size,
                            config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embed = self.embeding(x) # [batchsize, seqlen, embed_size]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).reshape(out.size()[0], -1)
        out = self.fc(out)
        out = self.softmax(out)
        return out

if __name__ == "__main__":
    from configs import Config
    cfg = Config()
    cfg.pad_size = 640
    model_textcls = Model(config=cfg)
    input_tensor = torch.tensor([i for i in range(640)]).reshape([1, 640])
    out_tensor = model_textcls.forward(input_tensor)
    print(out_tensor.size())
    print(out_tensor)