import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from transformers import (
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)

class PolicyGradientNetwork(nn.Module):

    def __init__(self, configs):
        super().__init__()
        # self.fc1 = nn.Linear(8, 16)
        # self.fc2 = nn.Linear(16, 16)
        # self.fc3 = nn.Linear(16, 4)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(configs.pretrained_model)

    def forward(self, inputs):
        # hid = torch.tanh(self.fc1(state))
        # hid = torch.tanh(self.fc2(hid))
        # return F.softmax(self.fc3(hid), dim=-1)
        output = self.model(**inputs)
        return F.softmax(output.logits, dim=2)

class PolicyGradientAgent():
    
    def __init__(self, network, configs):
        self.network = network
        self.tokenizer = BlenderbotTokenizer.from_pretrained(configs.pretrained_model)
        self.max_len = configs.max_len
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)
        self.device = configs.device
        
    def learn(self, log_probs: torch.Tensor, rewards: torch.Tensor) -> float:
        loss = (-log_probs * rewards).sum() # You don't need to revise this to pass simple baseline (but you can)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    def sample(self, state: str):
        inputs = {}
        inputs['input_ids'] = self.tokenizer(
            [state],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
        ).input_ids.to(self.device)
        inputs['decoder_input_ids'] = torch.tensor([self.tokenizer.start_token_id], device=self.device)
        action_prob = self.network(inputs)
        
        action_dist = Categorical(action_prob[0])
        action_ids = action_dist.sample()
        log_prob = torch.mean(action_dist.log_prob(action_ids))
        action = self.tokenizer.decode(action_ids, skip_special_tokens=True)

        return action, log_prob