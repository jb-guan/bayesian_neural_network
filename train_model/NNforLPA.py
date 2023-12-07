from bellotti_r_master_thesis.experiment_config import ExperimentConfig
from bellotti_r_master_thesis.awa_training import train_awa_forward_model,\
                                                  train_awa_invertible_model
from mllib.model import KerasSurrogate
import os
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

def absolute_percentage_error(y_true, y_pred, qoi_columns):
    error = (y_true - y_pred) / y_true * 100.
    error = pd.DataFrame(data=np.abs(error), columns=qoi_columns)
    return error

def updateFile(file,old_str,new_str):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = new_str+'\n'
            file_data += line
    with open(file,"w",encoding="utf-8") as f:
        f.write(file_data)

##################################################################################################################
def run(dff0,dff1,dff2,path,path_json='D:\guan\invertible_networks_for_optimisation_of_pc-master\data\\test_nn.json'):
    rangr = len(dff0)
    validation_fraction = 0.3/100
    training_fraction = (1. - validation_fraction*100)/100
    #path_length = []
    #for i in range(rangr):
     #   path_length.append(i*0.05)
    #df0 = pd.DataFrame(np.array(path_length).T,columns=['Path length'])

    #file_name = path + 'data\\df0.csv'
    #df0 = pd.read_csv(file_name, low_memory = False)

    #dff0['Path length'] = df0['Path length'][0:rangr]
    #dff2['Path length'] = df0['Path length'][0:rangr]
    validation_fraction = 0.3
    training_fraction = 1. - validation_fraction*100
    
    config = ExperimentConfig(path_json)

    # log info
    version = f'tensorflow version: {tf.__version__}'

    logging.basicConfig(filename=f'{config.save_dir}/logs/{config.experiment_name}.log',
                        level=logging.INFO)
    logging.info(version)

    logging.info('Name: ' + config.experiment_name)

    # path to save the trained model at
    model_dir = f'{config.save_dir}/models'

    # load the data
    rang = 3200
    val = 400
    X_train = dff0[['params_Q','params_em','params_length','params_pos','Path length']][0:rang]
    y_train = dff2[['energy','spread','length','radius','emittance','charge']][0:rang]
    X_val = dff0[['params_Q','params_em','params_length','params_pos','Path length']][rang:rang+val]
    y_val = dff2[['energy','spread','length','radius','emittance','charge']][rang:rang+val]

    # round the path length to make sure rounding errors don't affect the
    # groupby() in the evaluation
    X_train['Path length'] = np.round(X_train['Path length'], decimals=3)
    X_val['Path length'] = np.round(X_val['Path length'], decimals=3)

    #qoi_cols = df.columns

    #quantiles_to_log = [0., 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.]

    # path to log the losses and metrics to
    tensorboard_dir = f'{config.save_dir}/tensorboard_{config.experiment_name}'
    config.model_parameters['tensorboard_dir'] = tensorboard_dir

    batch_size = config.model_parameters['batch_size']
    #################################################################### train
    try:
        surr = train_awa_invertible_model(config.qoi_bounds,X_train,y_train,X_val,y_val,config.model_parameters,version,config.experiment_name)
    except:
        return None
    ########################################### forward prediction
    X_pre = dff0[['params_Q','params_em','params_length','params_pos','Path length']][rang+val:rangr]
    y_train_predict = surr.predict(X_pre.values,
                                        batch_size=batch_size)
    y_pre = dff2[['energy','spread','length','radius','emittance','charge','Path length']][rang+val:rangr]
    nl = ['energy','spread','length','radius','emittance','charge']
    R2_forward = []
    for value in range(len(nl)):
        ou = np.array(y_pre[nl[value]]).T
        y_pred = y_train_predict.T[value]
        y_a = sum(ou)/len(ou)
        SStot = sum((ou-y_a)**2)
        ei = []
        for j in range(len(ou)):
            ei.append((float(ou[j]-y_pred[j]))**2)
        SSres = sum(ei)
        R2 = 1-SSres/SStot
        R2_forward.append(R2)
    aver_for = sum(R2_forward)/len(R2_forward)
    with open("scan_para_nn.txt", "a+") as f:
        f.write('{} {}\n'.format(R2_forward,aver_for))
    ##########################################inn
    y_sample = dff2[['energy','spread','length','radius','emittance','charge','Path length']][rang+val:rangr]
    sample_train = surr.sample_n_tries(y_sample.values, batch_size=batch_size,n_tries=1000)
    yy = dff0[['params_Q','params_em','params_length','params_pos','Path length']][rang+val:rangr]
    nl_inn = ['params_Q','params_em','params_length','params_pos']
    R2_inn = []
    for value in range(len(nl_inn)):  #calculate R**2
        ou = np.array(yy[nl_inn[value]]).T
        y_pred = sample_train.T[value]
        y_a = sum(ou)/len(ou)
        SStot = sum((ou-y_a)**2)
        ei = []
        for j in range(len(ou)):
            ei.append((float(ou[j]-y_pred[j]))**2)
        SSres = sum(ei)
        R2 = 1-SSres/SStot
        R2_inn.append(R2)
    aver_inn = sum(R2_inn)/len(R2_inn)
    with open("scan_para_nn.txt", "a+") as f:  #save data
        f.write('{} {}\n'.format(R2_inn,aver_inn))
        f.write('\n')
    if aver_for>0.92 and aver_inn>0.92:  #save model
        save_path = path+'model{}'.format(aver_inn)
        os.mkdir(save_path)
        dff0.to_csv(save_path+'\dff0.csv')
        dff1.to_csv(save_path+'\dff1.csv')
        dff2.to_csv(save_path+'\dff2.csv')
        surr.save(save_path+'\mode5.h5')
        
if __name__ == '__main__':
    path = 'D:\\guan\\invertible_networks_for_optimisation_of_pc-master\\inn_test\\NNforLPA\\'
    ##################################################################################################################
    file_name0 = path + 'data\\dff0.csv'
    file_name1 = path + 'data\\dff1.csv'
    file_name2 = path + 'data\\dff2.csv'
    dff0 = pd.read_csv(file_name0, low_memory = False)#Preventing pop-up Warnings
    dff1 = pd.read_csv(file_name1, low_memory = False)#Preventing pop-up Warnings
    dff2 = pd.read_csv(file_name2, low_memory = False)#Preventing pop-up Warnings
    ##################################################################################################################
    z_dim = [1]#range(6,16)
    learning_rate = [0.005]
    batch_size = [30]
    number_of_blocks = [8]#range(4,17)
    coefficient_net_unit = [6]#range(2,9)
    path_json=path + 'jsonfile\\NNforLPA.json'
    #updateFile(file=path+'jsonfile\\test_model.json',old_str='training_data_paths',new_str='	"training_data_paths": ["{}\\data\\data.hdf5"],'.format(path))
    for i in z_dim:
        updateFile(file=path_json,old_str='z_dim',new_str='        "z_dim": {},'.format(i))
        for j in learning_rate:
            updateFile(file=path_json,old_str="learning_rate",new_str='        "learning_rate": {},'.format(j))
            for k in batch_size:
                updateFile(file=path_json,old_str="batch_size",new_str='        "batch_size": {},'.format(k))
                for m in number_of_blocks:
                    updateFile(file=path_json,old_str="number_of_blocks",new_str='        "number_of_blocks": {},'.format(m))
                    for n in coefficient_net_unit:
                        updateFile(file=path_json,old_str="coefficient_net_units",new_str='        "coefficient_net_units": [10,10,10,{}]'.format(n))
                        aa = [i,j,k,m,n]
                        with open("scan_para_nn.txt", "a+") as f:
                            f.write('{}\n'.format(aa))
                        run(dff0,dff1,dff2,path,path_json)
        

