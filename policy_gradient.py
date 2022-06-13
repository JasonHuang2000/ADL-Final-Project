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
        # output = self.model(input_ids=inputs['input_ids'], decoder_input_ids=inputs['decoder_input_ids'])
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

    def save_model(self, path: str):
        torch.save(self.network.state_dict(), path)

    def sample(self, state: str):
        inputs = {}
        inputs = self.tokenizer(
            [state],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
        ).to(self.device)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        decoder_input = []
        log_prob = []
        count = 0
        next_id = self.tokenizer.bos_token_id
        decoder_input.append(next_id)
        # decoder_input.append(inputs['decoder_input_ids'])
        # action_prob = self.network(inputs)
        while True :
            # print(decoder_input)
            inputs['decoder_input_ids'] = torch.tensor([decoder_input], device=self.device)
            action_prob = self.network(inputs)
            # print(action_prob.size())
            next_id = action_prob[0][count].argmax().item()
            log_prob.append(action_prob[0][count].max())
            decoder_input.append(next_id)
            # print(f"id {count}: {next_id} / {self.tokenizer.decode(next_id)}")
            count += 1
            if (count == self.max_len):
                break
            if (next_id == 2):
                break
        
        # print(self.tokenizer.decode(decoder_input))
        action = self.tokenizer.decode(decoder_input, skip_special_tokens=True)
        log_prob = sum(log_prob) / len(log_prob)

        # action_dist = Categorical(action_prob[0])
        # action_ids = action_dist.sample()
        # log_prob = torch.mean(action_dist.log_prob(action_ids))
        # action = self.tokenizer.decode(action_ids, skip_special_tokens=True)

        # print(f"action: {action}\nlog_prob: {log_prob}")
        return action, log_prob
