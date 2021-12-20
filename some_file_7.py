import numpy as np
import matplotlib.pyplot as plt

# open true values
with open("./npy/test_subset_my_idea.npy", mode="rb") as file:
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
plt.xlabel("ML preds")
plt.ylabel("NF preds")
plt.scatter(x=ml_preds[:, 0], y=nf_preds[:, 0])
plt.show()

plt.figure()
plt.title("log-nugget preds")
plt.xlabel("ML preds")
plt.ylabel("NV preds")
plt.scatter(x=ml_preds[:, 0], y=nv_preds[:, 0])
plt.show()


plt.figure()
plt.title("spatial-range preds")
plt.xlabel("ML preds")
plt.ylabel("NF preds")
plt.scatter(x=ml_preds[:, 1], y=nf_preds[:, 1])
plt.show()

plt.figure()
plt.title("spatial-range preds")
plt.xlabel("ML preds")
plt.ylabel("NV preds")
plt.scatter(x=ml_preds[:, 1], y=nv_preds[:, 1])
plt.show()

# scatter NN30 preds vs ML30 preds
plt.figure()
plt.title("log-nugget 30 preds")
plt.xlabel("ML30 preds")
plt.ylabel("NF30 preds")
plt.scatter(x=ml30_preds[:, 0], y=nf30_preds[:, 0])
plt.show()

plt.figure()
plt.title("log-nugget 30 preds")
plt.xlabel("ML30 preds")
plt.ylabel("NV30 preds")
plt.scatter(x=ml30_preds[:, 0], y=nv30_preds[:, 0])
plt.show()

plt.figure()
plt.title("spatial-range 30 preds")
plt.xlabel("ML30 preds")
plt.ylabel("NF30 preds")
plt.scatter(x=ml30_preds[:, 1], y=nf30_preds[:, 1])
plt.show()

plt.figure()
plt.title("spatial-range 30 preds")
plt.xlabel("ML30 preds")
plt.ylabel("NV30 preds")
plt.scatter(x=ml30_preds[:, 1], y=nv30_preds[:, 1])
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

nf_log_nugget_std = np.std(nf_true_difference[:, 0])
nf30_log_nugget_std = np.std(nf30_true_difference[:, 0])
nv_log_nugget_std = np.std(nv_true_difference[:, 0])
nv30_log_nugget_std = np.std(nv30_true_difference[:, 0])
ml_log_nugget_std = np.std(ml_true_difference[:, 0])
ml30_log_nugget_std = np.std(ml30_true_difference[:, 0])

nf_spatial_range_bias = np.average(nf_true_difference[:, 1])
nf30_spatial_range_bias = np.average(nf30_true_difference[:, 1])
nv_spatial_range_bias = np.average(nv_true_difference[:, 1])
nv30_spatial_range_bias = np.average(nv30_true_difference[:, 1])
ml_spatial_range_bias = np.average(ml_true_difference[:, 1])
ml30_spatial_range_bias = np.average(ml30_true_difference[:, 1])

nf_spatial_range_std = np.std(nf_true_difference[:, 1])
nf30_spatial_range_std = np.std(nf30_true_difference[:, 1])
nv_spatial_range_std = np.std(nv_true_difference[:, 1])
nv30_spatial_range_std = np.std(nv30_true_difference[:, 1])
ml_spatial_range_std = np.std(ml_true_difference[:, 1])
ml30_spatial_range_std = np.std(ml30_true_difference[:, 1])

# bias vs std plot
plt.figure()
plt.title("NN vs ML bias log-nugget")
plt.xlabel("bias")
plt.ylabel("standard deviation")
plt.scatter(x=nf_log_nugget_bias, y=nf_log_nugget_std, label="nf")
plt.scatter(x=nv_log_nugget_bias, y=nv_log_nugget_std, label="nv")
plt.scatter(x=ml_log_nugget_bias, y=ml_log_nugget_std, label="ml")
plt.legend()
plt.show()

plt.figure()
plt.title("NN vs ML bias spatial-range")
plt.xlabel("bias")
plt.ylabel("standard deviation")
plt.scatter(x=nf_spatial_range_bias, y=nf_spatial_range_std, label="nf")
plt.scatter(x=nv_spatial_range_bias, y=nv_spatial_range_std, label="nv")
plt.scatter(x=ml_spatial_range_bias, y=ml_spatial_range_std, label="ml")
plt.legend()
plt.show()

plt.figure()
plt.title("NN30 vs ML30 bias log-nugget")
plt.xlabel("bias 30")
plt.ylabel("standard deviation 30")
plt.scatter(x=nf30_log_nugget_bias, y=nf30_log_nugget_std, label="nf30")
plt.scatter(x=nv30_log_nugget_bias, y=nv30_log_nugget_std, label="nv30")
plt.scatter(x=ml30_log_nugget_bias, y=ml30_log_nugget_std, label="ml30")
plt.legend()
plt.show()

plt.figure()
plt.title("NN30 vs ML30 bias spatial-range")
plt.xlabel("bias 30")
plt.ylabel("standard deviation 30")
plt.scatter(x=nf30_spatial_range_bias, y=nf30_spatial_range_std, label="nf30")
plt.scatter(x=nv30_spatial_range_bias, y=nv30_spatial_range_std, label="nv30")
plt.scatter(x=ml30_spatial_range_bias, y=ml30_spatial_range_std, label="ml30")
plt.legend()
plt.show()


# compute preds - true_vals
nf_true_difference = nf_preds - true_vals
nf30_true_difference = nf30_preds - true_vals
nv_true_difference = nv_preds - true_vals
nv30_true_difference = nv30_preds - true_vals
ml_true_difference = ml_preds - true_vals
ml30_true_difference = ml30_preds - true_vals

# make box plots
plt.figure()
plt.title("Log-nugget box plots")
plt.ylabel("log-nugget preds - log-nugget")
plt.boxplot([nf_true_difference[:, 0], nv_true_difference[:, 0], ml_true_difference[:, 0]], labels=["NF", "NV", "ML"])
plt.show()

plt.figure()
plt.title("Spatial-range box plots")
plt.ylabel("spatial-range preds - spatial-range")
plt.boxplot([nf_true_difference[:, 1], nv_true_difference[:, 1], ml_true_difference[:, 1]], labels=["NF", "NV", "ML"])
plt.show()

plt.figure()
plt.title("Log-nugget box plots")
plt.ylabel("log-nugget preds - log-nugget")
plt.boxplot([nf30_true_difference[:, 0], nv30_true_difference[:, 0], ml30_true_difference[:, 0]],
            labels=["NF30", "NV30", "ML30"])
plt.show()

plt.figure()
plt.title("Spatial-range box plots")
plt.ylabel("spatial-range preds - spatial-range")
plt.boxplot([nf30_true_difference[:, 1], nv30_true_difference[:, 1], ml30_true_difference[:, 1]],
            labels=["NF30", "NV30", "ML30"])
plt.show()
