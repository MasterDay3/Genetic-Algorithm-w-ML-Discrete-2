"""File where you can test how much time you need for full GA cycle.
This is not very precise, but you can see the order of magnitude of working time."""

import time
import copy
import numpy as np
from models import fitness_function, DEFAULT_MODEL
from data import X_train, y_train

chromosome = np.ones(X_train.shape[1], dtype=int)
start = time.time()
fitness_function(
    chromosome, X_train.values, y_train.values, copy.deepcopy(DEFAULT_MODEL), cv=5
)
elapsed = time.time() - start

total = elapsed * 30 * 15 * 5  # N_GENERATION * CV * Population_size
print(f"Один fitness: {elapsed:.1f} сек")
print(f"Приблизний час: {total/60:.0f} хв")
