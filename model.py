from transformers import AutoTokenizer
import torch
from torch import nn
from transformers import EsmModel
from util import printl
import numpy as np
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def get_tokenizer(configs):
    if configs.model_scheme.lower() == 'esm2':
        tokenizer = AutoTokenizer.from_pretrained(configs.encoder_name)
    else:
        raise ValueError("Wrong model_scheme specified for get_tokenizer.")
    return tokenizer


def get_model(configs, log_path):
    if configs.model_scheme.lower() == 'esm2':
        encoder = Encoder(configs=configs, log_path=log_path)
    else:
        raise ValueError("Wrong model_scheme specified for get_model")
    return encoder


def get_esm(configs, log_path):
    model = EsmModel.from_pretrained(configs.encoder_name)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    if configs.training_mode.lower() == "frozen":
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        printl("All model parameters have been frozen.", log_path=log_path)
    elif configs.training_mode.lower() == "finetune":
        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.encoder.layer[configs.finetune_layer:].parameters():
            param.requires_grad = True
        printl(f"Parameters in the last {0 - configs.finetune_layer} layers are trainable.", log_path=log_path)
        for param in model.pooler.parameters():
            param.requires_grad = False
        printl("Pooling layer parameters have been frozen.", log_path=log_path)
    else:
        raise ValueError("Wrong training_mode specified for get_esm.")
    return model


def load_opt(model, config):
    scheduler = None
    if config.optimizer.name.lower() == 'adam':
        # opt = eval('torch.optim.' + config.optimizer.name)(model.parameters(), lr=config.optimizer.lr, eps=eps,
        #                                       weight_decay=config.optimizer.weight_decay)
        opt = torch.optim.AdamW(
            model.parameters(), lr=float(config.optimizer.lr),
            betas=(config.optimizer.beta_1, config.optimizer.beta_2),
            weight_decay=float(config.optimizer.weight_decay),
            eps=float(config.optimizer.eps)
        )
    elif config.optimizer.name.lower() == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=float(config.optimizer.lr),
                              momentum=0.9, dampening=0,
                              weight_decay=float(config.optimizer.weight_decay))

    else:
        raise ValueError('wrong optimizer')
    return opt, scheduler


def get_optimizer(net, configs, num_train_samples, logfilepath):
    optimizer, scheduler = load_opt(net, configs)
    if configs.optimizer.mode == "skip":
        scheduler = None
    else:
        if scheduler is None:
            if configs.optimizer.decay.first_cycle_steps:
                first_cycle_steps = configs.optimizer.decay.first_cycle_steps
            else:
                first_cycle_steps=np.ceil(
                    num_train_samples / configs.train_settings.grad_accumulation) * configs.train_settings.num_epochs / configs.optimizer.decay.num_restarts
            print("first_cycle_steps="+str(first_cycle_steps))
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=first_cycle_steps,
                cycle_mult=1.0,
                max_lr=configs.optimizer.lr,
                min_lr=configs.optimizer.decay.min_lr,
                warmup_steps=configs.optimizer.decay.warmup,
                gamma=configs.optimizer.decay.gamma)
    return optimizer, scheduler


class Encoder(nn.Module):
    def __init__(self, configs, log_path):
        super().__init__()
        self.model = get_esm(configs, log_path)
        # self.pooling_layer = nn.AdaptiveAvgPool2d((None, 1))
        # self.pooling_layer = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return x


class LayerNormNet(nn.Module):
    """
    This implementation was inspired by https://github.com/tttianhao/CLEAN
    """
    def __init__(self, configs, hidden_dim=512, out_dim=256):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = configs.supcon.drop_out
        self.device = configs.train_settings.device
        self.dtype = torch.float32
        feature_dim = {"facebook/esm2_t6_8M_UR50D": 320, "facebook/esm2_t33_650M_UR50D": 1280,
                       "facebook/esm2_t30_150M_UR50D": 640, "facebook/esm2_t12_35M_UR50D": 480}
        self.fc1 = nn.Linear(feature_dim[configs.encoder.model_name], hidden_dim, dtype=self.dtype, device=self.device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=self.dtype, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=self.dtype, device=self.device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=self.dtype, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=self.dtype, device=self.device)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
