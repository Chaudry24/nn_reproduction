import numpy as np
import matplotlib.pyplot as plt

with open("./results/training_loss.npy", mode="rb") as file:
    training_loss = np.load(file)

with open("./results/predictions_NN.npy", mode="rb") as file:
    preds_NN = np.load(file)

with open("./results/predictions_MLE.npy", mode="rb") as file:
    preds_MLE = np.load(file)

with open("./data/training_params.npy", mode="rb") as file:
    true_vals = np.load(file)


plt.figure()
plt.plot(training_loss[19900:20000])
plt.xlabel("Epochs")
plt.ylabel("training loss")

plt.figure()
plt.scatter(x=preds_NN[:, 0], y=preds_MLE[:, 0])
plt.xlabel("NN preds for spatial range")
plt.ylabel("MLE preds for spatial range")

plt.figure()
plt.scatter(x=preds_NN[:, 1], y=preds_MLE[:, 1])
plt.xlabel("NN preds for smoothness")
plt.ylabel("MLE preds for smoothness")

bias_NN_spatial_range = np.abs(np.average(preds_NN[:, 0] - true_vals[:, 0]))
bias_NN_smoothness = np.abs(np.average(preds_NN[:, 1] - true_vals[:, 1]))
bias_MLE_spatial_range = np.abs(np.average(preds_MLE[:, 0] - true_vals[:, 0]))
bias_MLE_smoothness = np.abs(np.average(preds_MLE[:, 1] - true_vals[:, 1]))

plt.figure()
plt.scatter(x=bias_NN_spatial_range, y=bias_MLE_spatial_range, c="r")
plt.scatter(x=bias_NN_smoothness, y=bias_MLE_smoothness, c="b")
