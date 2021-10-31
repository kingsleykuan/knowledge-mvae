from torch import nn
from transformers import BertModel

from .Optimus.code.pytorch_transformers import GPT2ForLatentConnector


# Updated implementation based on Hugging Face BERT
# Mimics behaviour of BertForLatentConnector
# Line 639 Optimus/code/pytorch_transformers/modeling_bert.py
class OptimusBert(BertModel):
    def __init__(self, config, latent_size=32):
        super(OptimusBert, self).__init__(config)

        self.linear = nn.Linear(
            config.hidden_size, 2 * latent_size, bias=False)

        self.init_weights()


# Wrapper around original implementation
class OptimusGPT2(GPT2ForLatentConnector):
    pass
