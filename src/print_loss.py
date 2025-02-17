import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("runs/experiment_6/performance.csv")

# print train loss and validation loss for each epoch
for i in range(len(data)):
    print(f"Epoch {data['Epoch'][i]}: Train Loss: {data['Train Loss'][i]}, Val Loss: {data['Val Loss'][i]}")

# print the last row with {column name} and {value} in detail
print(data.iloc[-1].to_dict())

# print train loss and validation loss for each epoch & MIoU
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(data['Epoch'], data['Train Loss'], label="Train Loss")
ax[0].plot(data['Epoch'], data['Val Loss'], label="Val Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Train and Validation Loss")
ax[0].legend()

ax[1].plot(data['Epoch'], data['Mean IoU'])
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("MIoU")
ax[1].set_title("Mean IoU")
plt.show()
