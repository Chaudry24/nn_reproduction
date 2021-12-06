import numpy as np
import matplotlib.pyplot as plt

# open true values
with open("./npy/test_my_idea.npy", mode="rb") as file:
    true_vals = np.load(file)

# open NN loss
with open("./my_idea/NF/training_loss_NF.npy", mode="rb") as file:
    nf_loss = np.load(file)
with open("./my_idea/NF30/training_loss_NF30.npy", mode="rb") as file:
    nf30_loss = np.load(file)
with open("./my_idea/NV/training_loss_NV.npy", mode="rb") as file:
    nv_loss = np.load(file)
with open("./my_idea/NV30/training_loss_NV30.npy", mode="rb") as file:
    nv30_loss = np.load(file)

# open NN preds
with open("./my_idea/NF/preds_NF.npy", mode="rb") as file:
    nf_preds = np.load(file)
with open("./my_idea/NF30/preds_NF30.npy", mode="rb") as file:
    nf30_preds = np.load(file)
with open("./my_idea/NV/preds_NV.npy", mode="rb") as file:
    nv_preds = np.load(file)
with open("./my_idea/NV30/preds_NV30.npy", mode="rb") as file:
    nv30_preds = np.load(file)

# open MLE preds
with open("./my_idea/ML/preds_MLE.npy", mode="rb") as file:
    ml_preds = np.load(file)
with open("./my_idea/ML30/preds_ML30.npy", mode="rb") as file:
    ml30_preds = np.load(file)

# plot NN loss
plt.figure()
plt.title("NN Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot(nf_loss, label="nf_loss")
plt.plot(nf30_loss, label="nf30_loss")
plt.plot(nv_loss, label="nv_loss")
plt.plot(nv30_loss, label="nv30_loss")
plt.legend()
plt.show()

# scatter NN preds vs ML preds
plt.figure()
plt.title("log-nugget preds")
plt.xlabel("NN preds")
plt.ylabel("ML preds")
plt.scatter(x=nf_preds[:, 0], y=ml_preds[:, 0], label="nf-ml log-nugget")
plt.scatter(x=nv_preds[:, 0], y=ml_preds[:, 0], label="nv-ml log-nugget")
plt.legend()
plt.show()

plt.figure()
plt.title("spatial-range preds")
plt.xlabel("NN preds")
plt.ylabel("ML preds")
plt.scatter(x=nf_preds[:, 1], y=ml_preds[:, 1], label="nf-ml spatial-range")
plt.scatter(x=nv_preds[:, 1], y=ml_preds[:, 1], label="nv-ml spatial range")
plt.legend()
plt.show()

# scatter NN30 preds vs ML30 preds
plt.figure()
plt.title("log-nugget 30 preds")
plt.xlabel("NN30 preds")
plt.ylabel("ML30 preds")
plt.scatter(x=nf30_preds[:, 0], y=ml30_preds[:, 0], label="nf30-ml30 log-nugget")
plt.scatter(x=nv30_preds[:, 0], y=ml30_preds[:, 0], label="nv30-ml30 log-nugget")
plt.legend()
plt.show()

plt.figure()
plt.title("spatial-range 30 preds")
plt.xlabel("NN30 preds")
plt.ylabel("ML30 preds")
plt.scatter(x=nf30_preds[:, 1], y=ml30_preds[:, 1], label="nf30-ml30 spatial-range")
plt.scatter(x=nv30_preds[:, 1], y=ml30_preds[:, 1], label="nv30-ml30 spatial-range")
plt.legend()
plt.show()

# bias
nf_true_difference = np.abs(true_vals-nf_preds)
nf30_true_difference = np.abs(true_vals-nf30_preds)
nv_true_difference = np.abs(true_vals-nv_preds)
nv30_true_difference = np.abs(true_vals-nv30_preds)
ml_true_difference = np.abs(true_vals-ml_preds)
ml30_true_difference = np.abs(true_vals-ml30_preds)

nf_log_nugget_bias = np.average(nf_true_difference[:, 0])
nf30_log_nugget_bias = np.average(nf30_true_difference[:, 0])
nv_log_nugget_bias = np.average(nv_true_difference[:, 0])
nv30_log_nugget_bias = np.average(nv30_true_difference[:, 0])
ml_log_nugget_bias = np.average(ml_true_difference[:, 0])
ml30_log_nugget_bias = np.average(ml30_true_difference[:, 0])

nf_spatial_range_bias = np.average(nf_true_difference[:, 1])
nf30_spatial_range_bias = np.average(nf30_true_difference[:, 1])
nv_spatial_range_bias = np.average(nv_true_difference[:, 1])
nv30_spatial_range_bias = np.average(nv30_true_difference[:, 1])
ml_spatial_range_bias = np.average(ml_true_difference[:, 1])
ml30_spatial_range_bias = np.average(ml30_true_difference[:, 1])

# bias plot
plt.figure()
plt.title("NN vs ML bias log-nugget")
plt.xlabel("NN bias")
plt.ylabel("ML bias")
plt.scatter(x=nf_log_nugget_bias, y=ml_log_nugget_bias, label="nf-ml log nugget bias")
plt.scatter(x=nv_log_nugget_bias, y=ml_log_nugget_bias, label="nv-ml log nugget bias")
plt.legend()
plt.show()

plt.figure()
plt.title("NN vs ML bias spatial-range")
plt.xlabel("NN bias")
plt.ylabel("ML bias")
plt.scatter(x=nf_spatial_range_bias, y=ml_spatial_range_bias, label="nf-ml spatial range bias")
plt.scatter(x=nv_spatial_range_bias, y=ml_spatial_range_bias, label="nv-ml spatial range bias")
plt.legend()
plt.show()

plt.figure()
plt.title("NN30 vs ML30 bias log-nugget")
plt.xlabel("NN30 bias")
plt.ylabel("ML30 bias")
plt.scatter(x=nf30_log_nugget_bias, y=ml30_log_nugget_bias, label="nf30-ml30 log nugget bias")
plt.scatter(x=nv30_log_nugget_bias, y=ml30_log_nugget_bias, label="nv30-ml30 log nugget bias")
plt.legend()
plt.show()

plt.figure()
plt.title("NN30 vs ML30 bias spatial-range")
plt.xlabel("NN30 bias")
plt.ylabel("ML30 bias")
plt.scatter(x=nf30_spatial_range_bias, y=ml30_spatial_range_bias, label="nf30-ml30 spatial range bias")
plt.scatter(x=nv30_spatial_range_bias, y=ml30_spatial_range_bias, label="nv30-ml30 spatial range bias")
plt.legend()
plt.show()
