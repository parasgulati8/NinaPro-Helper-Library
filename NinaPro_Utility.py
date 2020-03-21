import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import os
from tensorflow.keras.models import Sequential, Model, load_model
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras as K

def get_data(path,file):
    mat = loadmat(os.path.join(path,file))
    data = pd.DataFrame(mat['emg'])
    data['stimulus'] = mat['restimulus'] 
    data['repetition'] = mat['repetition']
    
    return data

def normalise(data, train_reps):
    x = [np.where(data.values[:,13] == rep) for rep in train_reps]
    indices = np.squeeze(np.concatenate(x, axis = -1))
    train_data = data.iloc[indices, :]
    train_data = data.reset_index(drop=True)
    
    scaler = StandardScaler(with_mean=True,
                                with_std=True,
                                copy=False).fit(train_data.iloc[:, :12])
    
    scaled = scaler.transform(data.iloc[:,:12])
    normalised = pd.DataFrame(scaled)
    normalised['stimulus'] = data['stimulus']
    normalised['repetition'] = data['repetition']
    return normalised

def filter_data(data, f, butterworth_order = 4, btype = 'lowpass'):
    emg_data = data.values[:,:12]
    
    f_sampling = 2000
    nyquist = f_sampling/2
    if isinstance(f, int):
        fc = f/nyquist
    else:
        fc = list(f)
        for i in range(len(f)):
            fc[i] = fc[i]/nyquist
            
    b,a = signal.butter(butterworth_order, fc, btype=btype)
    transpose = emg_data.T.copy()
    
    for i in range(len(transpose)):
        transpose[i] = (signal.lfilter(b, a, transpose[i]))
    
    filtered = pd.DataFrame(transpose.T)
    filtered['stimulus'] = data['stimulus']
    filtered['repetition'] = data['repetition']
    
    return filtered

def rectify(data):
    return abs(data)

def windowing(data, reps, gestures, win_len, win_stride):
    if reps:
        x = [np.where(data.values[:,13] == rep) for rep in reps]
        indices = np.squeeze(np.concatenate(x, axis = -1))
        data = data.iloc[indices, :]
        data = data.reset_index(drop=True)
        
    if gestures:
        x = [np.where(data.values[:,12] == move) for move in gestures]
        indices = np.squeeze(np.concatenate(x, axis = -1))
        data = data.iloc[indices, :]
        data = data.reset_index(drop=True)
        
    idx=  [i for i in range(win_len, len(data), win_stride)]
    
    X = np.zeros([len(idx), win_len, len(data.columns)-2])
    y = np.zeros([len(idx), ])
    reps = np.zeros([len(idx), ])
    
    for i,end in enumerate(idx):
        start = end - win_len
        X[i] = data.iloc[start:end, 0:12].values
        y[i] = data.iloc[end, 12]
        reps[i] = data.iloc[end, 13]
        
    return X, y, reps

def train_model(model, X_train_wind, y_train_wind, X_test_wind, y_test_wind, save_to, epoch = 300):
        from tensorflow import keras as K
        opt_adam = K.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy' , optimizer=opt_adam, metrics=['categorical_accuracy'])

#         log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
        mc = ModelCheckpoint(save_to + '_best_model.h5', monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)

        history = model.fit(x=X_train_wind, y=y_train_wind, epochs=epoch, shuffle=True,
                    verbose=1,
                    validation_data = (X_test_wind, y_test_wind), callbacks=[es, mc])

        saved_model = load_model(save_to + '_best_model.h5')
        # evaluate the model
        _, train_acc = saved_model.evaluate(X_train_wind, y_train_wind, verbose=0)
        _, test_acc = saved_model.evaluate(X_test_wind, y_test_wind, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        
        return history,saved_model

def get_categorical(y):
    return pd.get_dummies(pd.Series(y)).values

def plot_cnf_matrix(saved_model, X_valid_cv, target):    
    y_pred = saved_model.predict(X_valid_cv)
    model_predictions = [list(y_pred[i]).index(y_pred[i].max())+1 for i in range(len(y_pred))]

    conf_mx = confusion_matrix(target, model_predictions)
    plt.matshow(conf_mx)
    plt.show()

