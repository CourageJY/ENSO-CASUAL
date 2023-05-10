import numpy as np
import sys
from sklearn.model_selection import train_test_split

sys.path.append("")
from model.AutoEncoder.auto_encoder import *
from model.params import *

#从npz文件中加载numpy数据
sst = np.load(f"{params.final_data_dir}/sst-final.npz")['sst']
uwind = np.load(f"{params.final_data_dir}/uwind-final.npz")['uwind']
vwind = np.load(f"{params.final_data_dir}/vwind-final.npz")['vwind']
vapor = np.load(f"{params.final_data_dir}/vapor-final.npz")['vapor']
cloud = np.load(f"{params.final_data_dir}/cloud-final.npz")['cloud']
rain = np.load(f"{params.final_data_dir}/rain-final.npz")['rain']

#get train and test data
sst_train,sst_test=train_test_split(sst, test_size=params.train_eval_split, random_state=params.random_seed)
uwind_train,uwind_test=train_test_split(uwind, test_size=params.train_eval_split, random_state=params.random_seed)
vwind_train,vwind_test=train_test_split(vwind, test_size=params.train_eval_split, random_state=params.random_seed)
vapor_train,vapor_test=train_test_split(vapor, test_size=params.train_eval_split, random_state=params.random_seed)
cloud_train,cloud_test=train_test_split(cloud, test_size=params.train_eval_split, random_state=params.random_seed)
rain_train,rain_test=train_test_split(rain, test_size=params.train_eval_split, random_state=params.random_seed)

#merge
orgin=[sst,uwind,vwind,vapor,cloud,rain]
train=[sst_train,uwind_train,vwind_train,vapor_train,cloud_train,rain_train]
test=[sst_test,uwind_test,vwind_test,vapor_test,cloud_test,rain_test]

for i in range(len(train)):
    print("-------------------------")
    print("---------Step{}----------".format(i+1))
    print("-------------------------")
    autoencoder=Autoencoder(params.latent_dim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(train[i], train[i],
                    epochs=params.num_epochs,
                    batch_size=params.batch_size,
                    shuffle=True,
                    validation_data=(test[i], test[i]))
    
    #save the encoded data 
    encode_data=autoencoder.encoder(orgin[i]).numpy()
    np.savez(f'{params.encoder_save_dir}/{params.remote_sensing_variables[i]}-encoder.npz', **{params.remote_sensing_variables[i]: encode_data})

    # Save the weights
    autoencoder.save_weights(f'{params.encoder_save_dir}/{params.remote_sensing_variables[i]}-model')


print("end")
