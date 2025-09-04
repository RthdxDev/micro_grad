import torch
import torch.nn as nn_torch
from micro_grad_engine import Scalar
from nn import Sequential, Layer, Tanh, ReLU

def test_sanity_check():
    x = Scalar(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.item == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()
    print('Good')

def test_more_ops():
    a = Scalar(-4.0)
    b = Scalar(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.item - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
    print('Good')

def test_sequential_training():
    """
    Тестирует Sequential модель с Tanh, обучая её на матричном датасете и сравнивая с PyTorch
    """
    # Матричный датасет для тестирования
    inputs = [
        [1.0, 2.0, -0.5],
        [2.0, -1.0, 0.8], 
        [-0.5, 1.5, 2.0],
        [0.8, -0.3, 1.2]
    ]
    targets = [0.7, -0.2, 0.9, 0.1]
    
    print("\nTesting Sequential with Tanh on matrix dataset...")
    
    # Создание простой модели для надежного тестирования
    model_mg = Sequential(
        Layer(3, 2),
        Tanh(),
        Layer(2, 1)
    )
    
    # Создание аналогичной модели в PyTorch
    model_pt = nn_torch.Sequential(
        nn_torch.Linear(3, 2),
        nn_torch.Tanh(),
        nn_torch.Linear(2, 1)
    )
    
    # Устанавливаем фиксированные веса для точного сравнения
    with torch.no_grad():
        # Устанавливаем простые фиксированные веса в PyTorch
        model_pt[0].weight.data = torch.tensor([[0.5, -0.3, 0.2], [0.1, 0.7, -0.4]], dtype=torch.float64)
        model_pt[0].bias.data = torch.tensor([0.1, -0.1], dtype=torch.float64)
        model_pt[2].weight.data = torch.tensor([[0.8, -0.6]], dtype=torch.float64)
        model_pt[2].bias.data = torch.tensor([0.2], dtype=torch.float64)
    
    # Копируем те же веса в нашу модель
    layer1 = model_mg.modules[0]  # type: ignore
    layer1.neurons[0].w[0].item = 0.5  # type: ignore
    layer1.neurons[0].w[1].item = -0.3  # type: ignore  
    layer1.neurons[0].w[2].item = 0.2  # type: ignore
    layer1.neurons[0].b.item = 0.1  # type: ignore
    layer1.neurons[1].w[0].item = 0.1  # type: ignore
    layer1.neurons[1].w[1].item = 0.7  # type: ignore
    layer1.neurons[1].w[2].item = -0.4  # type: ignore
    layer1.neurons[1].b.item = -0.1  # type: ignore
    
    layer2 = model_mg.modules[2]  # type: ignore
    layer2.neurons[0].w[0].item = 0.8  # type: ignore
    layer2.neurons[0].w[1].item = -0.6  # type: ignore
    layer2.neurons[0].b.item = 0.2  # type: ignore
    
    learning_rate = 0.02
    epochs = 25
    
    print("Training both models...")
    
    for epoch in range(epochs):
        total_loss_mg = 0.0
        total_loss_pt = 0.0
        
        # Обучение нашей модели
        for inp, target_val in zip(inputs, targets):
            x_mg = [Scalar(val) for val in inp]
            y_pred_mg = model_mg(x_mg)[0]
            target_mg = Scalar(target_val)
            
            loss_mg = (y_pred_mg - target_mg) ** 2
            total_loss_mg += loss_mg.item
            
            model_mg.zero_grad()
            loss_mg.backward()
            
            for p in model_mg.parameters:
                p.item -= learning_rate * p.grad
        
        # Обучение PyTorch модели
        for inp, target_val in zip(inputs, targets):
            x_pt = torch.tensor([inp], dtype=torch.float64)
            y_pred_pt = model_pt(x_pt)[0]
            target_pt = torch.tensor([target_val], dtype=torch.float64)
            
            loss_pt = (y_pred_pt - target_pt) ** 2
            total_loss_pt += loss_pt.item()
            
            model_pt.zero_grad()
            loss_pt.backward()
            
            with torch.no_grad():
                for p in model_pt.parameters():
                    if p.grad is not None:
                        p.data = p.data - learning_rate * p.grad
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: MicroGrad Loss = {total_loss_mg:.6f}, PyTorch Loss = {total_loss_pt:.6f}")
    
    # Финальное сравнение на всех примерах
    print(f"\nFinal predictions:")
    tol = 1e-5
    max_diff = 0.0
    
    for i, (inp, target_val) in enumerate(zip(inputs, targets)):
        x_mg = [Scalar(val) for val in inp]
        y_pred_mg = model_mg(x_mg)[0]
        
        x_pt = torch.tensor([inp], dtype=torch.float64)
        y_pred_pt = model_pt(x_pt)[0]
        
        mg_result = y_pred_mg.item
        pt_result = y_pred_pt.item()
        diff = abs(mg_result - pt_result)
        max_diff = max(max_diff, diff)
        
        print(f"Sample {i+1}: MicroGrad = {mg_result:.6f}, PyTorch = {pt_result:.6f}, Diff = {diff:.8f}")
    
    print(f"Maximum difference: {max_diff:.8f}")
    assert max_diff < tol, f"Results differ too much: max_diff={max_diff} > {tol}"
    print("Sequential Tanh model test passed! ✓")

def test_sequential_with_relu():
    """
    Тестирует Sequential модель с ReLU активацией на матричном датасете, сравнивая с PyTorch
    """
    # Матричный датасет для тестирования
    inputs = [
        [1.5, -0.8, 2.0],
        [-1.2, 1.7, 0.4],
        [0.9, 2.3, -1.1],
        [2.1, -0.5, 0.6]
    ]
    targets = [0.6, -0.3, 0.8, 0.2]
    
    print("\nTesting Sequential with ReLU on matrix dataset...")
    
    # Создание модели с ReLU активацией
    model_mg = Sequential(
        Layer(3, 2),
        ReLU(),
        Layer(2, 1)
    )
    
    # Создание аналогичной модели в PyTorch
    model_pt = nn_torch.Sequential(
        nn_torch.Linear(3, 2),
        nn_torch.ReLU(),
        nn_torch.Linear(2, 1)
    )
    
    # Устанавливаем фиксированные веса для точного сравнения
    with torch.no_grad():
        # Первый слой (3->2)
        model_pt[0].weight.data = torch.tensor([
            [0.3, -0.4, 0.6], 
            [0.5, 0.2, -0.3]
        ], dtype=torch.float64)
        model_pt[0].bias.data = torch.tensor([0.1, -0.2], dtype=torch.float64)
        
        # Второй слой (2->1)
        model_pt[2].weight.data = torch.tensor([[0.7, -0.8]], dtype=torch.float64)
        model_pt[2].bias.data = torch.tensor([0.3], dtype=torch.float64)
    
    # Копируем те же веса в нашу модель
    # Первый слой (3->2)
    layer1 = model_mg.modules[0]  # type: ignore
    layer1.neurons[0].w[0].item = 0.3  # type: ignore
    layer1.neurons[0].w[1].item = -0.4  # type: ignore
    layer1.neurons[0].w[2].item = 0.6  # type: ignore
    layer1.neurons[0].b.item = 0.1  # type: ignore
    
    layer1.neurons[1].w[0].item = 0.5  # type: ignore
    layer1.neurons[1].w[1].item = 0.2  # type: ignore
    layer1.neurons[1].w[2].item = -0.3  # type: ignore
    layer1.neurons[1].b.item = -0.2  # type: ignore
    
    # Второй слой (2->1)
    layer2 = model_mg.modules[2]  # type: ignore
    layer2.neurons[0].w[0].item = 0.7  # type: ignore
    layer2.neurons[0].w[1].item = -0.8  # type: ignore
    layer2.neurons[0].b.item = 0.3  # type: ignore
    
    learning_rate = 0.01
    epochs = 20
    
    print("Training both ReLU models...")
    
    for epoch in range(epochs):
        total_loss_mg = 0.0
        total_loss_pt = 0.0
        
        # Обучение нашей модели
        for inp, target_val in zip(inputs, targets):
            x_mg = [Scalar(val) for val in inp]
            y_pred_mg = model_mg(x_mg)[0]
            target_mg = Scalar(target_val)
            
            loss_mg = (y_pred_mg - target_mg) ** 2
            total_loss_mg += loss_mg.item
            
            model_mg.zero_grad()
            loss_mg.backward()
            
            for p in model_mg.parameters:
                p.item -= learning_rate * p.grad
        
        # Обучение PyTorch модели
        for inp, target_val in zip(inputs, targets):
            x_pt = torch.tensor([inp], dtype=torch.float64)
            y_pred_pt = model_pt(x_pt)[0]
            target_pt = torch.tensor([target_val], dtype=torch.float64)
            
            loss_pt = (y_pred_pt - target_pt) ** 2
            total_loss_pt += loss_pt.item()
            
            model_pt.zero_grad()
            loss_pt.backward()
            
            with torch.no_grad():
                for p in model_pt.parameters():
                    if p.grad is not None:
                        p.data = p.data - learning_rate * p.grad
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: MicroGrad Loss = {total_loss_mg:.6f}, PyTorch Loss = {total_loss_pt:.6f}")
    
    # Финальное сравнение на всех примерах
    print(f"\nFinal ReLU predictions:")
    tol = 1e-5
    max_diff = 0.0
    
    for i, (inp, target_val) in enumerate(zip(inputs, targets)):
        x_mg = [Scalar(val) for val in inp]
        y_pred_mg = model_mg(x_mg)[0]
        
        x_pt = torch.tensor([inp], dtype=torch.float64)
        y_pred_pt = model_pt(x_pt)[0]
        
        mg_result = y_pred_mg.item
        pt_result = y_pred_pt.item()
        diff = abs(mg_result - pt_result)
        max_diff = max(max_diff, diff)
        
        print(f"Sample {i+1}: MicroGrad = {mg_result:.6f}, PyTorch = {pt_result:.6f}, Diff = {diff:.8f}")
    
    print(f"Maximum difference: {max_diff:.8f}")
    assert max_diff < tol, f"Results differ too much: max_diff={max_diff} > {tol}"
    print("Sequential ReLU model test passed! ✓")

if __name__ == "__main__":
    # Запускаем все тесты
    test_sanity_check()
    test_more_ops()
    test_sequential_training()
    test_sequential_with_relu()
    print("\nAll tests passed successfully! 🎉")
