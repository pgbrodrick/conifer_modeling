

import pandas as pd
from tqdm import tqdm
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D, Concatenate, SeparableConv2D, Dropout
import keras
import numpy as np

import matplotlib as mpl
mpl.use('Agg')


def main():
    # Manage datasets
    data_set = pd.read_csv('data/cover_extraction.csv')

    xy = np.array(data_set[['X_UTM', 'Y_UTM']])
    shade = np.array(data_set['ered_B_1']).flatten()
    #tch = np.array(data_set['_tch_B_1']).flatten()
    refl = np.array(data_set)[:, -427:-1].astype(np.float32)

    
    covertype = np.array(data_set[['covertype']]).flatten()


    aspen = np.zeros(len(xy)).astype(bool)
    aspen[covertype == 'aspen'] = True

    bad_bands_refl = np.zeros(426).astype(bool)
    bad_bands_refl[:8] = True
    bad_bands_refl[192:205] = True
    bad_bands_refl[284:327] = True
    bad_bands_refl[417:] = True
    refl[:, bad_bands_refl] = np.nan
    refl = refl[:, np.all(np.isnan(refl) == False, axis=0)]

    good_data = np.ones(len(xy)).astype(bool)
    good_data = good_data.flatten()

    refl = refl[good_data, ...]
    aspen = aspen[good_data, ...]
    xy = xy[good_data, :]

    Y = (aspen == True).reshape(-1, 1)

    np.random.seed(13)
    perm = np.random.permutation(Y.shape[0])
    train = np.zeros(perm.shape).astype(bool)

    n_xstep = 100
    n_ystep = 100
    x_space = np.linspace(np.min(xy[:, 0]), np.max(xy[:, 0]), n_xstep)
    y_space = np.linspace(np.min(xy[:, 1]), np.max(xy[:, 1]), n_ystep)
    grids = np.zeros((len(y_space), len(x_space))).flatten()
    grids[np.random.permutation(len(grids))[:int(0.85*len(grids))]] = 1
    grids = grids.reshape((len(y_space), len(x_space)))
    for n in tqdm(range(0, n_xstep-1), ncols=80):
        for m in range(0, n_ystep-1):
            if(grids[m, n] == 1):
                valid = np.logical_and(xy[:, 0] > x_space[n], xy[:, 0] <= x_space[n+1])
                valid[xy[:, 1] <= y_space[m]] = False
                valid[xy[:, 1] > y_space[m+1]] = False
                train[valid] = True

    test = np.logical_not(train)
    print((np.sum(train) / float(len(train))))
    print((np.sum(test) / float(len(train))))

    weights = np.zeros(Y.shape[0])
    needles_w = float(len(Y[train]))/float(np.sum(Y == 1))**0.8
    noneedles_w = float(len(Y[train]))/float(np.sum(Y == 0))**0.8
    print('needles_weight: {}'.format(needles_w/(needles_w+noneedles_w)))
    print('noneedles_weight: {}'.format(noneedles_w/(needles_w+noneedles_w)))
    weights[Y.flatten() == 1] = needles_w / (needles_w+noneedles_w) * 100
    weights[Y.flatten() == 0] = noneedles_w / (needles_w+noneedles_w) * 100

    # Scale by brightness, and save the scaler
    brightness = np.sqrt(np.mean(np.power(refl, 2), axis=-1))
    refl = refl / brightness[:, np.newaxis]
    scaler = preprocessing.StandardScaler()
    scaler.fit(refl[train, :])
    refl = scaler.transform(refl)
    joblib.dump(scaler, 'trained_models/nn_aspen_scaler')

    """
    # Train a random forest - not ultimately used, so commented out, but was checked
    model = RandomForestClassifier(n_estimators=200, max_depth=5, n_jobs=20, random_state=13)
    model.fit(refl[train,:],Y[train],weights[train])
    
    print ('RF Results')
    pred = model.predict(refl).reshape(-1,1)
    train_cf = confusion_matrix(Y[train],pred[train])
    test_cf = confusion_matrix(Y[test],pred[test])
    train_cf = train_cf / np.sum(train_cf,axis=1)[:,np.newaxis]
    test_cf = test_cf / np.sum(test_cf,axis=1)[:,np.newaxis]
    
    print('Train CF: {} {}'.format(round(train_cf[0,0],2),round(train_cf[1,1],2)))
    print('Test CF: {} {}'.format(round(test_cf[0,0],2),round(test_cf[1,1],2)))
    
    
    # Train an SVM - no ultimately used, so commented out, but was checked
    from sklearn.svm import SVC
    model = SVC(gamma='auto',kernel='poly',degree=5)
    model.fit(refl[train,:],Y[train],weights[train])
    
    print ('SVM Results')
    pred = model.predict(refl).reshape(-1,1)
    train_cf = confusion_matrix(Y[train],pred[train])
    test_cf = confusion_matrix(Y[test],pred[test])
    train_cf = train_cf / np.sum(train_cf,axis=1)[:,np.newaxis]
    test_cf = test_cf / np.sum(test_cf,axis=1)[:,np.newaxis]
    
    print('Train CF: {} {}'.format(round(train_cf[0,0],2),round(train_cf[1,1],2)))
    print('Test CF: {} {}'.format(round(test_cf[0,0],2),round(test_cf[1,1],2)))
    """

    # Create NN model structure, and compile.  Ultimate structure desided after pretty
    # extensive (heuristically guided) testing
    inlayer = keras.layers.Input(shape=(refl.shape[1],))
    output_layer = inlayer
    output_layer = Dense(units=400)(output_layer)
    output_layer = keras.layers.LeakyReLU(alpha=0.3)(output_layer)
    output_layer = Dropout(0.4)(output_layer)
    output_layer = BatchNormalization()(output_layer)

    output_layer = Dense(units=400)(output_layer)
    output_layer = keras.layers.LeakyReLU(alpha=0.3)(output_layer)
    output_layer = Dropout(0.4)(output_layer)
    output_layer = BatchNormalization()(output_layer)

    output_layer = Dense(units=400)(output_layer)
    output_layer = keras.layers.LeakyReLU(alpha=0.3)(output_layer)
    output_layer = Dropout(0.4)(output_layer)
    output_layer = BatchNormalization()(output_layer)

    output_layer = Dense(units=400)(output_layer)
    output_layer = keras.layers.LeakyReLU(alpha=0.3)(output_layer)
    output_layer = Dropout(0.4)(output_layer)
    output_layer = BatchNormalization()(output_layer)

    output_layer = Dense(units=2, activation='sigmoid')(output_layer)
    model = keras.models.Model(inputs=[inlayer], outputs=[output_layer])
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # reformat Y data for crossentropy loss function
    cY = np.hstack([Y, 1-Y])

    # Run through and train model in 5 epoch steps - basically instituiting a manual stoping criteria because loss is really a function of both TPR and FPRp, and
    # we wanted to select a good combination of both.
    print('NN Results')
    for e in range(30):
        model.fit(refl[train, ...], cY[train, ...], epochs=5, sample_weight=weights[train],
                  validation_data=(refl[test, ...], cY[test, ...], weights[test]), batch_size=1000)

        pred = model.predict(refl)
        pred = np.argmin(pred, axis=1)

        train_cf = confusion_matrix(Y[train], pred[train])
        test_cf = confusion_matrix(Y[test], pred[test])
        print('TPR,FPRp,TNR:\n{} {} {}'.format(np.round(train_cf[0, 0]/np.sum(train_cf[0, :]), 3),
                                               np.round(train_cf[0, 1]/np.sum(train_cf[1, :]), 3),
                                               np.round(train_cf[1, 1]/np.sum(train_cf[1, :]), 3)))

        print('{} {} {}\n'.format(np.round(test_cf[0, 0]/np.sum(test_cf[0, :]), 3),
                                  np.round(test_cf[0, 1]/np.sum(test_cf[1, :]), 3),
                                  np.round(test_cf[1, 1]/np.sum(test_cf[1, :]), 3)))
        print('Precision, Recall::\n{} {}'.format(np.round(train_cf[0,0]/np.sum(train_cf[:,0]),3),
                                                  np.round(train_cf[0,0]/np.sum(train_cf[0,:]),3)))

        print('{} {}'.format(np.round(test_cf[0,0]/np.sum(test_cf[:,0]),3),
                             np.round(test_cf[0,0]/np.sum(test_cf[0,:]),3)))

        model.save('trained_models/aspen_nn_set_{}.h5'.format(e))


if __name__ == "__main__":
    main()
