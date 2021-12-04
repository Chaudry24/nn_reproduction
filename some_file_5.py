import numpy as np
import matplotlib.pyplot as plt

# open true values
with open("./npy/test_subset.npy", mode="rb") as file:
    true_vals = np.load(file)

# open NN loss
with open("./tf_stat_reproduction/NF/training_loss_NF.npy", mode="rb") as file:
    nf_loss = np.load(file)
with open("./tf_stat_reproduction/NF30/training_loss_NF30.npy", mode="rb") as file:
    nf30_loss = np.load(file)
with open("./tf_stat_reproduction/NV/training_loss_NV.npy", mode="rb") as file:
    nv_loss = np.load(file)
with open("./tf_stat_reproduction/NV30/training_loss_NV30.npy", mode="rb") as file:
    nv30_loss = np.load(file)

# open NN preds
with open("./tf_stat_reproduction/NF/preds_NF.npy", mode="rb") as file:
    nf_preds = np.load(file)
with open("./tf_stat_reproduction/NF30/preds_NF30.npy", mode="rb") as file:
    nf30_preds = np.load(file)
with open("./tf_stat_reproduction/NV/preds_NV.npy", mode="rb") as file:
    nv_preds = np.load(file)
with open("./tf_stat_reproduction/NV30/preds_NV30.npy", mode="rb") as file:
    nv30_preds = np.load(file)

# open MLE preds
with open("./tf_stat_reproduction/ML/preds_MLE.npy", mode="rb") as file:
    ml_preds = np.load(file)
with open("./tf_stat_reproduction/ML30/preds_ML30.npy", mode="rb") as file:
    ml30_preds = np.load(file)

# plot NN loss
plt.figure()
plt.plot(nf_loss, label="nf_loss")
plt.plot(nf30_loss, label="nf30_loss")
plt.plot(nv_loss, label="nv_loss")
plt.plot(nv30_loss, label="nv30_loss")
plt.legend()
plt.show()

# scatter NN preds vs ML preds
plt.figure()
plt.title("log-nugget preds")
plt.scatter(x=nf_preds[:, 0], y=ml_preds[:, 0])
plt.xlabel("nf preds")
plt.ylabel("ml preds")
plt.figure()
plt.title("log-nugget preds")
plt.scatter(x=nv_preds[:, 0], y=ml_preds[:, 0])
plt.xlabel("nv preds")
plt.ylabel("ml preds")
plt.figure()
plt.title("spatial-range preds")
plt.scatter(x=nf_preds[:, 1], y=ml_preds[:, 1])
plt.xlabel("nf preds")
plt.ylabel("ml preds")
plt.figure()
plt.title("spatial-range preds")
plt.scatter(x=nv_preds[:, 1], y=ml_preds[:, 1])
plt.xlabel("nv preds")
plt.ylabel("ml preds")
plt.show()

# scatter NN30 preds vs ML30 preds
plt.figure()
plt.title("log-nugget 30 preds")
plt.scatter(x=nf30_preds[:, 0], y=ml30_preds[:, 0])
plt.xlabel("nf30 preds")
plt.ylabel("ml30 preds")
plt.figure()
plt.title("log-nugget 30 preds")
plt.scatter(x=nv30_preds[:, 0], y=ml30_preds[:, 0])
plt.xlabel("nv30 preds")
plt.ylabel("ml30 preds")
plt.figure()
plt.title("spatial-range 30 preds")
plt.scatter(x=nf30_preds[:, 1], y=ml30_preds[:, 1])
plt.xlabel("nf30 preds")
plt.ylabel("ml30 preds")
plt.figure()
plt.title("spatial-range 30 preds")
plt.scatter(x=nv30_preds[:, 1], y=ml30_preds[:, 1])
plt.xlabel("nv30 preds")
plt.ylabel("ml30 preds")
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
plt.title("nf vs ml bias log-nugget")
plt.scatter(x=nf_log_nugget_bias, y=ml_log_nugget_bias, label="nf-ml log nugget bias")
plt.xlabel("nf bias")
plt.ylabel("ml bias")

plt.title("nv vs ml bias log-nugget")
plt.scatter(x=nv_log_nugget_bias, y=ml_log_nugget_bias, label="nv-ml log nugget bias")
plt.xlabel("nv bias")
plt.ylabel("ml bias")

plt.title("nf vs ml bias spatial-range")
plt.scatter(x=nf_spatial_range_bias, y=ml_spatial_range_bias, label="nf-ml spatial range bias")
plt.xlabel("nf bias")
plt.ylabel("ml bias")

plt.title("nv vs ml bias spatial-range")
plt.scatter(x=nv_spatial_range_bias, y=ml_spatial_range_bias, label="nv-ml spatial range bias")
plt.xlabel("nv bias")
plt.ylabel("ml bias")

plt.legend()
plt.show()

plt.figure()
plt.title("nf30 vs ml30 bias log-nugget")
plt.scatter(x=nf30_log_nugget_bias, y=ml30_log_nugget_bias, label="nf30-ml30 log nugget bias")
plt.xlabel("nf30 bias")
plt.ylabel("ml30 bias")

plt.title("nv30 vs ml30 bias log-nugget")
plt.scatter(x=nv30_log_nugget_bias, y=ml30_log_nugget_bias, label="nv30-ml30 log nugget bias")
plt.xlabel("nv30 bias")
plt.ylabel("ml30 bias")

plt.title("nf30 vs ml30 bias spatial-range")
plt.scatter(x=nf30_spatial_range_bias, y=ml30_spatial_range_bias, label="nf30-ml30 spatial range bias")
plt.xlabel("nf30 bias")
plt.ylabel("ml30 bias")

plt.title("nv30 vs ml30 bias spatial-range")
plt.scatter(x=nv30_spatial_range_bias, y=ml30_spatial_range_bias, label="nv30-ml30 spatial range bias")
plt.xlabel("nv30 bias")
plt.ylabel("ml30 bias")

plt.legend()
plt.show()
