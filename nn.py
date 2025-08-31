import random
from abc import ABC, abstractmethod
from micro_grad_engine import Scalar


class Module(ABC):
    def __init__(self) -> None:
        self.parameters = []

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
        return self._parameters
    
    @parameters.setter
    def parameters(self, params: list[Scalar]) -> None:
        self._parameters = params


class Neuron(Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(input_size)]
        self.b = Scalar(0)
        self.parameters = self.w + [self.b]

    def forward(self, x) -> list[Scalar]:
        out: Scalar = sum((x_i * w_i for x_i, w_i in zip(x, self.w)), self.b)
        return [out]
    
    def __repr__(self) -> str:
        return f"Neuron(w={self.w[:3]}{'...' if len(self.w) > 3 else ''}, b={self.b})"


class Layer(Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.neurons: list[Neuron] = [Neuron(input_size) for _ in range(output_size)]
        self.parameters = [p for neuron in self.neurons for p in neuron.parameters]

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
        self.parameters = [p for module in self.modules for p in module.parameters]

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
