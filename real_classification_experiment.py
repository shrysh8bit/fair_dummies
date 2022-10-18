

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

dataset = "adult"

# use minibatch sgd
batch_size = 32

# step size to minimize loss
lr_loss = 0.5

# step size used to fit binary classifier (discriminator)
lr_dis = 0.5

# pre-train discriminator and predictive model
pre_training_steps = 2

# inner epochs to fit loss
loss_steps = 1

# inner epochs to fit discriminator classifier
dis_steps = 1

# equalized odds penalty
mu_val = 0.9
second_scale = 0.00001


# total number of epochs
epochs = 50

# utility loss
cost_pred = torch.nn.CrossEntropyLoss()

# base predictive model
model_type = "adult_model"
# model_type = "deep_model"
# dataset = "nursery"





## Load data


print("dataset: " + dataset)

X, A, Y, X_cal, A_cal, Y_cal, X_test, A_test, Y_test = get_dataset.get_train_test_data(base_path, dataset, seed)

in_shape = X.shape[1]
num_classes = len(np.unique(Y))

print("training samples (used to fit predictive model) = " + str(X.shape[0]) + " p = " + str(X.shape[1]))
print("holdout samples (used to fit fair-dummies test statistics function) = " + str(X_cal.shape[0]))
print("test samples = " + str(X_test.shape[0]))



model = fair_dummies_learning.EquiClassLearner(lr=lr_loss,
                                               pretrain_pred_epochs=pre_training_steps,
                                               pretrain_dis_epochs=pre_training_steps,
                                               epochs=epochs,
                                               loss_steps=loss_steps,
                                               dis_steps=dis_steps,
                                               cost_pred=cost_pred,
                                               in_shape=in_shape,
                                               batch_size=batch_size,
                                               model_type=model_type,
                                               lambda_vec=mu_val,
                                               second_moment_scaling=second_scale,
                                               num_classes=num_classes,
                                               save_folder = "runs/real_classification_exp")

input_data_train = np.concatenate((A[:,np.newaxis],X),1)
print(f"--> input data train l 96 dimns  {input_data_train.shape} , Y {Y.shape}")
if(dataset == 'adult'):
    Y = Y.reshape(18096,)
print(f"--> input data train l 96 dimns  {input_data_train.shape} , Y {Y.shape}")
model.fit(input_data_train, Y)




input_data_cal = np.concatenate((A_cal[:,np.newaxis],X_cal),1)
Yhat_out_cal = model.predict(input_data_cal)

input_data_test = np.concatenate((A_test[:,np.newaxis],X_test),1)
Yhat_out_test = model.predict(input_data_test)


misclassification = 1.0-utility_functions.compute_acc_numpy(Yhat_out_test, Y_test)
print("misclassification error = " + str(misclassification))




p_val = utility_functions.fair_dummies_test_classification(Yhat_out_cal,
                                                           A_cal,
                                                           Y_cal,
                                                           Yhat_out_test,
                                                           A_test,
                                                           Y_test,
                                                           num_reps = 1,
                                                           num_p_val_rep=1000,
                                                           reg_func_name="Net")






