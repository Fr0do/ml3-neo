import numpy as np

def test_submission_shape():
    # Мы предполагаем, что submission лежит в homework/02-pytorch/submission/predictions.npy
    # Для теста можно создать mock, или проверять сам файл если он есть.
    # Этот тест в основном запускается локально студентом.
    
    try:
        preds = np.load("homework/02-pytorch/submission/predictions.npy")
    except FileNotFoundError:
        return # Пропускаем, если файла еще нет
        
    assert len(preds.shape) == 1, f"Shape must be (N,), got {preds.shape}"
    assert preds.dtype in [np.int64, np.int32], f"Dtype must be integer, got {preds.dtype}"
    assert preds.min() >= 0 and preds.max() <= 9, "Predictions must be in range [0, 9] for CIFAR-10"
