import pandas as pd
import numpy as np
PRICEDOMSIZE=  5  # domain size of prices
SIZEDOMSIZE= 100
def createTable(N):
    return pd.DataFrame({
            'pA': np.random.randint(0, PRICEDOMSIZE, N),
            'pB': np.random.randint(0, PRICEDOMSIZE, N),
            'sA': np.random.randint(0, SIZEDOMSIZE, N),
            'sB': np.random.randint(0, SIZEDOMSIZE, N)})
createTable(5)