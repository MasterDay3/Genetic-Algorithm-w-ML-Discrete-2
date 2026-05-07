import time
import numpy as np
import copy
from models import fitness_function, DEFAULT_MODEL
from data import X_train, y_train

# rom main import N_GENERATION

chromosome = np.ones(X_train.shape[1], dtype=int)
start = time.time()
fitness_function(
    chromosome, X_train.values, y_train.values, copy.deepcopy(DEFAULT_MODEL), cv=5
)
elapsed = time.time() - start

total = elapsed * 30 * 15 * 5  # 5 - розмір популяції з main.py
print(f"Один fitness: {elapsed:.1f} сек")
print(f"Приблизний час: {total/60:.0f} хв")
