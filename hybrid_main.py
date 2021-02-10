import torch

def fn(x):
    return torch.abs(2*x)
    
traced_fn = torch.jit.trace(fn, torch.rand(()))

@torch.jit.script
def script_fn(x):
    z = torch.ones([1], dtype=torch.int64)
    for i in range(int(x)):
        z = z * (i + 1)
    return z
    
class TracedModule(torch.nn.Module):
    def forward(self, x):
        x = x.type(torch.float32)
        return torch.floor(torch.sqrt(x) / 5.)
        
class ScriptModule(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x):
        r = -x
        if int(torch.fmod(x, 2.0)) == 0.0:
            r = x / 2.0
        return r

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.traced_module = torch.jit.trace(TracedModule(), torch.rand(()))
        self.script_module = ScriptModule()

        print('traced_fn graph', traced_fn.graph)
        print('script_fn graph', script_fn.graph)
        print('TracedModule graph', self.traced_module.__getattr__('forward').graph)
        print('ScriptModule graph', self.script_module.__getattr__('forward').graph)

    def forward(self, x):
        x = traced_fn(x)
        x = script_fn(x)

        x = self.traced_module(x)
        x = self.script_module(x)

        return x

n = Net()
print(n(torch.tensor([5]))) 

n_traced = torch.jit.trace(n, torch.tensor([5]))
print(n_traced(torch.tensor([5])))
print('n_traced graph', n_traced.graph)