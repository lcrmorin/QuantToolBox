# SVM classification (hard margin)

Translated from `Examples/svm/svm1.m`. 15 points in 2-D, two linearly
separable classes.

```python
import numpy as np
from quanttoolbox.svm.svm import svm_classification_primal

data = np.array([
    [0.5, 2.5, 1], [2.7, 4.2, 1], [2.7, 2.0, 1], [1.7, 4.2, 1], [1.5, 0.7, 1],
    [2.3, 5.3, 1], [4.0, 6.9, 1],
    [6.4, 4.5, -1], [7.7, 2.2, -1], [8.8, 6.0, -1], [7.4, 6.5, -1],
    [6.5, 1.7, -1], [8.3, 1.3, -1], [6.0, 1.3, -1], [5.0, 0.5, -1],
])
y, x = data[:, 2], data[:, 0:2]

result = svm_classification_primal(y, x, c=None)  # c=None -> hard margin
print("beta0:", round(result.beta0, 6))
print("beta:", np.round(result.beta, 6))
print("margin:", round(result.margin, 6))

# recover support vectors: points exactly on the margin (residual ~ 0)
residuals = y - result.beta0 - x @ result.beta
support_vectors = np.where(np.abs(residuals) < 1e-5)[0]
print("support vector indices:", support_vectors)
```

Output:

```text
beta0: 2.415929
beta: [-0.707965  0.247788]
margin: 1.777422
support vector indices: [ 2  7 14]
```

Only 3 of the 15 points end up as support vectors — the rest are strictly
inside their class's margin and don't constrain the separating
hyperplane at all, which is the hallmark sparsity property of the SVM
solution.
