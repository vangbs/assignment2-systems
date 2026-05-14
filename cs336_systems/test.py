import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        print(self.fc1.weight.dtype, self.fc2.weight.dtype, self.ln.weight.dtype)
        x = self.relu(self.fc1(x))
        print(x.dtype)
        x = self.ln(x)
        print('ln:', x.dtype)
        x = self.fc2(x)
        print(x.dtype)
        return x

if __name__ == "__main__":
    model = ToyModel(10, 10).to("cuda")
    x = torch.randn(10, 10).to("cuda")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x)
        loss = y.pow(2).sum()
        print(loss.dtype)
    
    loss.backward()
    print(model.fc1.weight.grad.dtype, model.fc2.weight.grad.dtype, model.ln.weight.grad.dtype)