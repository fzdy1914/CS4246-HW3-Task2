import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    def __init__(self, buffer_limit=5000):
        self.buffer = []
        self.buffer_limit = buffer_limit

    def push(self, transition):
        if len(self.buffer) == self.buffer_limit:
            self.buffer.pop(0)
            self.buffer.append(transition)
        else:
            self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.choices(self.buffer, k=batch_size)
        torch_samples = (
            torch.stack([torch.FloatTensor(s[0]).to(device) for s in samples]),
            torch.stack([torch.LongTensor(s[1]).to(device) for s in samples]),
            torch.stack([torch.FloatTensor(s[2]).to(device) for s in samples]),
            torch.stack([torch.FloatTensor(s[3]).to(device) for s in samples]),
            torch.stack([torch.FloatTensor(s[4]).to(device) for s in samples])
        )
        return torch_samples

    def __len__(self):
        return len(self.buffer)
