




import os
import sys

base_path = os.getcwd() + '/data/'

import torch
import random
import get_dataset
import numpy as np
import pandas as pd
from fair_dummies import utility_functions
from fair_dummies import fair_dummies_learning

seed = 123




random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

dataset = "crimes"

# use all data in sgd (without minibatches)
batch_size = 10000

# step size to minimize loss
lr_loss = 0.01

# step size used to fit binary classifier (discriminator)
lr_dis = 0.01

# inner epochs to fit loss
loss_steps = 80

# inner epochs to fit discriminator classifier
dis_steps = 80

# equalized odds penalty
mu_val = 0.7
second_scale = 1


# total number of epochs
epochs = 20

# utility loss
cost_pred = torch.nn.MSELoss()

# base predictive model
model_type = "linear_model"






## Load data


print("dataset: " + dataset)

X, A, Y, X_cal, A_cal, Y_cal, X_test, A_test, Y_test = get_dataset.get_train_test_data(base_path, dataset, seed)

in_shape = X.shape[1]

print("training samples (used to fit predictive model) = " + str(X.shape[0]) + " p = " + str(X.shape[1]))
print("holdout samples (used to fit fair-dummies test statistics function) = " + str(X_cal.shape[0]))
print("test samples = " + str(X_test.shape[0]))






model = fair_dummies_learning.EquiRegLearner(lr=lr_loss,
                                             pretrain_pred_epochs=0,
                                             pretrain_dis_epochs=0,
                                             epochs=epochs,
                                             loss_steps=loss_steps,
                                             dis_steps=dis_steps,
                                             cost_pred=cost_pred,
                                             in_shape=in_shape,
                                             batch_size=batch_size,
                                             model_type=model_type,
                                             lambda_vec=mu_val,
                                             second_moment_scaling=second_scale,
                                             out_shape=1)

input_data_train = np.concatenate((A[:,np.newaxis],X),1)
model.fit(input_data_train, Y)





input_data_cal = np.concatenate((A_cal[:,np.newaxis],X_cal),1)
Yhat_out_cal = model.predict(input_data_cal)

input_data_test = np.concatenate((A_test[:,np.newaxis],X_test),1)
Yhat_out_test = model.predict(input_data_test)






rmse_trivial = np.sqrt(np.mean((np.mean(Y_test)-Y_test)**2))
print("RMSE trivial = " + str(rmse_trivial))

rmse = np.sqrt(np.mean((Yhat_out_test-Y_test)**2))
print("RMSE trained model = " + str(rmse))

p_val = utility_functions.fair_dummies_test_regression(Yhat_out_cal,
                                                       A_cal,
                                                       Y_cal,
                                                       Yhat_out_test,
                                                       A_test,
                                                       Y_test,
                                                       num_reps = 1,
                                                       num_p_val_rep=1000,
                                                       reg_func_name="Net")





