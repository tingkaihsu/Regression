#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import awkward as ak
import uproot_methods


# In[2]:


import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


# In[3]:


def stack_arrays(a, keys, axis=-1):
    flat_arr = np.stack([a[k].flatten() for k in keys], axis=axis)
    return awkward.JaggedArray.fromcounts(a[keys[0]].counts, flat_arr)


# In[4]:


def pad_array(a, maxlen, value=0., dtype='float32'):
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x


# In[5]:


##and Professor suggests that we could use mass, classifacation for later application
def SetAKArr(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    n_particles_ls = []
    px_ls = []
    py_ls = []
    pz_ls = []
    energy_ls = []
    mass_ls = []
    charge_ls = []
    _label1 = []
    _label2 = []
    _label3 = []
    _label4 = []
    _label5 = []
    
    n = 0
    for line in lines:
        if line.startswith('E'):
            if not n == 0:
                n_particles_ls.append(n)
                n = 0
            exp_inf = line.split()
#             _label1.append(int(exp_inf[1]))
#             _label2.append(1-int(exp_inf[1]))
#             _label1.append(1)
#             _label2.append(0)
            _label1.append(float(exp_inf[1]))
            _label2.append(float(exp_inf[2]))
            _label3.append(float(exp_inf[3]))
            _label4.append(float(exp_inf[4]))
            _label5.append(float(exp_inf[5]))
        else:
            par = line.split()
            ##particle +1
            n = n + 1
            px_ls.append(float(par[2]))
            py_ls.append(float(par[3]))
            pz_ls.append(float(par[4]))
            energy_ls.append(float(par[5]))
            mass_ls.append(float(par[6]))
            charge_ls.append(int(par[0]))
#             px_ls.append(6)
#             py_ls.append(2)
#             pz_ls.append(3)
#             energy_ls.append(4)
#             mass_ls.append(5)
    if not n == 0:
        n_particles_ls.append(n)
    px_arr = np.array(px_ls)
    py_arr = np.array(py_ls)
    pz_arr = np.array(pz_ls)
    energy_arr = np.array(energy_ls)
    mass_arr = np.array(mass_ls)
    charge_arr = np.array(charge_ls)
    n_particles = np.array(n_particles_ls)

#     print(n_particles)
    px = ak.JaggedArray.fromcounts(n_particles, px_arr)
    py = ak.JaggedArray.fromcounts(n_particles, py_arr)
    pz = ak.JaggedArray.fromcounts(n_particles, pz_arr)
    energy = ak.JaggedArray.fromcounts(n_particles, energy_arr)
    mass = ak.JaggedArray.fromcounts(n_particles, mass_arr)
    charge = ak.JaggedArray.fromcounts(n_particles, charge_arr)
    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, energy)
    ##Create an Order Dic
    from collections import OrderedDict
    v = OrderedDict()
    v['part_px'] = px
#     print(px)
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy
    v['part_mass'] = mass
    v['charge'] = charge
#     v['part_e_log'] = np.log(energy)
#     v['part_px_log'] = np.log(px)
#     v['part_py_log'] = np.log(py)
#     v['part_pz_log'] = np.log(pz)
#     v['part_m_log'] = np.log(mass)
    v['label'] = np.stack((_label1, _label2, _label3, _label4, _label5), axis = -1)
    v['label'] = np.stack(_label5, axis = -1)
#     print(v['label'])
    return v


# In[ ]:





# In[6]:


class Dataset(object):
    def __init__(self, filepath, feature_dict = {}, label = 'label', pad_len=100, data_format='channel_first'):
        self.filepath = filepath
        self.feature_dict = feature_dict
        if len(feature_dict) == 0:
            feature_dict['points'] = ['part_px','part_py','part_pz']
            feature_dict['features'] = ['part_energy', 'part_mass', 'charge', 'part_px', 'part_py', 'part_pz']
            feature_dict['mask'] = ['part_energy']
        ##currently we use 'E' for experiments
        self.label = label
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values = {}
        self._label = None
        self._load()
        
    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
#         counts = None
        a = SetAKArr(self.filepath)
        self._label = a[self.label]
        for k in self.feature_dict:
                cols = self.feature_dict[k]
                if not isinstance(cols, (list, tuple)):
                    cols = [cols]
                arrs = []
                for col in cols:
                    arrs.append(pad_array(a[col], self.pad_len))
                    ##check the dimesion of a[col], and it should be array.
                self._values[k] = np.stack(arrs, axis=self.stack_axis)
        logging.info('Finished loading file %s' % self.filepath)
        
        
        
    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key==self.label:
            return self._label
        else:
            return self._values[key]
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]


# In[7]:


train_dataset = Dataset('train.txt', data_format='channel_last')
val_dataset = Dataset('val.txt', data_format='channel_last')
test_dataset = Dataset('test.txt', data_format = 'channel_last')


# In[ ]:





# In[8]:


import tensorflow as tf
from tensorflow import keras
from tf_keras_model import get_particle_net, get_particle_net_lite


# In[9]:


model_type = 'particle_net_lite' # choose between 'particle_net' and 'particle_net_lite'
##this shows the number of classes for classification
try:
    num_classes = train_dataset.y.shape[1]
except:
    num_classes = 1
input_shapes = {k:train_dataset[k].shape[1:] for k in train_dataset.X}
if 'lite' in model_type:
    model = get_particle_net_lite(num_classes, input_shapes)
else:
    model = get_particle_net(num_classes, input_shapes)


# In[10]:


# Training parameters
batch_size = 1024 if 'lite' in model_type else 384
epochs = 30


# In[11]:


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr *= 0.1
    elif epoch > 20:
        lr *= 0.01
    logging.info('Learning rate: %f'%lr)
    return lr


# In[12]:


# model.compile(loss='categorical_crossentropy',
#               optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
#               metrics=['accuracy'])
# model.compile(loss='log_cosh',
#               optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
#               metrics=['accuracy'])
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()


# In[13]:


from tensorflow.keras.callbacks import Callback
class LossLogger(Callback):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
#         self.lb = lb

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a') as f:
#             print("Epoch ", epoch + 1,": loss = ", logs["val_loss"], "\n", file = f)
#             if (epoch+1)%5==0 or epoch==0:
            print('V ', logs['val_loss'], file = f)
            print('L', logs['loss'], file = f)
#             print()
#             f.write()
# loss_logger = LossLogger('MSE_vac_loss.txt')
loss_logger = LossLogger('MSE_loss.txt')


# In[14]:


# Prepare model model saving directory.
import os
save_dir = 'model_checkpoints'
model_name = '%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = keras.callbacks.ModelCheckpoint(filepath='loss.txt',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
# I change the monitor from val_acc to val_loss
# checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
#                              monitor='val_loss',
#                              verbose=1,
#                              save_best_only=True)
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
progress_bar = keras.callbacks.ProgbarLogger()
callbacks = [checkpoint, lr_scheduler, loss_logger]
# callbacks = [lr_schedule]


# In[15]:

#make model
train_dataset.shuffle()
model.fit(train_dataset.X, train_dataset.y,
          batch_size=batch_size,
#           epochs=epochs,
          epochs=5,
          validation_data=(val_dataset.X, val_dataset.y),
          shuffle=True,
          callbacks=callbacks)
# In[16]:

##print mass
with open('MSE_pre_40epoches.txt', 'w') as file:
    
    predictions = model.predict(test_dataset.X)
    for prediction in predictions:
        print(prediction, file = file)    
