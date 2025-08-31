import math

class Scalar:
    def __init__(self, item, prev=(), op='', grad=0) -> None:
        self.item = item
        self.grad = grad
        self._prev = set(prev)
        self._op = op
        self._backward = lambda: None

    def __add__(self, other) -> 'Scalar':
        assert isinstance(other, (int, float, Scalar)), "Scalar must be a number"
        other = Scalar(other) if not isinstance(other, Scalar) else other
        out = Scalar(self.item + other.item, (self, other), op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other) -> 'Scalar':
        assert isinstance(other, (int, float, Scalar)), "Scalar must be a number"
        other = Scalar(other) if not isinstance(other, Scalar) else other
        out = Scalar(self.item + other.item, (self, other), op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> 'Scalar':
        assert isinstance(other, (int, float, Scalar)), "Scalar must be a number"
        other = Scalar(other) if not isinstance(other, Scalar) else other
        out = Scalar(self.item * other.item, (self, other), op='*')

        def _backward():
            self.grad += out.grad * other.item
            other.grad += out.grad * self.item

        out._backward = _backward
        return out

    def __rmul__(self, other) -> 'Scalar':
        assert isinstance(other, (int, float, Scalar)), "Scalar must be a number"
        other = Scalar(other) if not isinstance(other, Scalar) else other
        out = Scalar(self.item * other.item, (self, other), op='*')

        def _backward():
            self.grad += out.grad * other.item
            other.grad += out.grad * self.item

        out._backward = _backward
        return out
    
    def __pow__(self, power) -> 'Scalar':
        assert isinstance(power, (int, float)), "Power must be a int/float"
        out = Scalar(self.item ** power, (self, ), op='**')

        def _backward():
            self.grad += out.grad * power * (self.item ** (power - 1))

        out._backward = _backward
        return out
    
    def __neg__(self) -> 'Scalar':
        out = self * -(1.0)
        return out

    def __sub__(self, other) -> 'Scalar':
        return self + -other
    
    def __truediv__(self, other) -> 'Scalar':
        return self * other ** (-1)
    
    def backward(self):
        topo = []
        visited = set()
        
        def dfs(v):
            if v not in visited:
                visited.add(v)
                for u in v._prev:
                    dfs(u)
                topo.append(v)
        
        dfs(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Scalar({self.item, self.grad})"

