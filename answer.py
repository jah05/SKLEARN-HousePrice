from houseprice import cleanData
from pandas import pd
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

data = pd.read_csv("test.csv")
print(data)
