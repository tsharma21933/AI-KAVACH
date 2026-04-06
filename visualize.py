import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset.csv")

data.hist(figsize=(10,10))
plt.show()