import random
from abc import ABC, abstractmethod
from micro_grad_engine import Scalar

class Module(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, x) -> list[Scalar]:
        pass

    def __call__(self, x) -> list[Scalar]:
        return self.forward(x)

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = 0

    @property
    def parameters(self) -> list[Scalar]:
        return []


class Neuron(Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(input_size)]
        self.b = Scalar(0)
    
    @property
    def parameters(self) -> list[Scalar]:
        return self.w + [self.b]

    def forward(self, x) -> list[Scalar]:
        out: Scalar = sum((x_i * w_i for x_i, w_i in zip(x, self.w)), self.b)
        return [out]
    
    def __repr__(self) -> str:
        return f"Neuron(w={self.w[:3]}{'...' if len(self.w) > 3 else ''}, b={self.b})"


class Layer(Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.neurons: list[Neuron] = [Neuron(input_size) for _ in range(output_size)]

    @property
    def parameters(self) -> list[Scalar]:
        return [param for neuron in self.neurons for param in neuron.parameters]

    def forward(self, x) -> list[Scalar]:
        return [neuron(x)[0] for neuron in self.neurons]
    
    def __len__(self) -> int:
        return len(self.neurons)

    def __repr__(self) -> str:
        out = f"Layer("
        out += "\n".join(repr(neuron) for neuron in self.neurons[:5])
        if len(self.neurons) > 5:
            out += "\n..."
        out += ")"
        return out


class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()
        self.modules = args

    @property
    def parameters(self) -> list[Scalar]:
        return [p for module in self.modules for p in module.parameters]

    def forward(self, x) -> list[Scalar]:
        for module in self.modules:
            x = module(x)
        return x
    
    def __repr__(self) -> str:
        out = f"Sequential("
        for module in self.modules:
            out += repr(module)
            if isinstance(module, Layer):
                out += f'\nNeurons: {len(module)}'
            out += '\n\n'
        out += ")"
        return out
    
class Tanh(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> list[Scalar]:
        return [xi.tanh() for xi in x]

    def __repr__(self) -> str:
        return "Tanh()"
    

class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> list[Scalar]:
        return [xi.relu() for xi in x]

    def __repr__(self) -> str:
        return "ReLU()"