import pandas as pd

data = pd.read_csv("runs/experiment_2/performance.csv")

# print train loss and validation loss for each epoch
for i in range(len(data)):
    print(f"Epoch {data['Epoch'][i]}: Train Loss: {data['Train Loss'][i]}, Val Loss: {data['Val Loss'][i]}")

# print the last row with {column name} and {value} in detail
print(data.iloc[-1].to_dict())