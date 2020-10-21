

import numpy as np
import pandas as pd
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D, Concatenate, SeparableConv2D, Dropout



#mpl.use('Agg')


def main():

    layers_range = [2, 4, 6]
    node_range = [100, 250, 400]
    dropout_range = [0.2, 0.4]
    refl, Y, test, train, weights, covertype = manage_datasets('~/Google Drive File Stream/My Drive/CB_share/NEON/cover_classification/extraction_output/cover_extraction.csv')

    for cover in np.unique(covertype):
        fraction = np.sum(covertype == cover) / len(covertype)
        print(f'{cover}: {fraction}')
    quit()
    #preds, dim_names = nn_scenario_test(refl, Y, weights, test, train, n_epochs=2, its=100, layers_range=layers_range,
    #                                    node_range=node_range, dropout_range=dropout_range)
    plotting_model_fits_singleclass(Y, test, covertype, layers_range, node_range, dropout_range)


def manage_datasets(import_data, category='aspen', weighting=False):
    # Import extraction csv
    data_set = pd.read_csv(import_data)

    # extract x y coordinates for each pixel
    xy = np.array(data_set[['X_UTM', 'Y_UTM']])
    #shade = np.array(data_set['ered_B_1']).flatten()
    #tch = np.array(data_set['_tch_B_1']).flatten()

    # extract reflectance data from csv
    refl = np.array(data_set)[:, -427:-1].astype(np.float32)

    # Extract cover type data
    covertype = np.array(data_set[['covertype']]).flatten()
    print(np.unique(covertype))

    # Identifying aspen in boolian vector format
    aspen = np.zeros(len(xy)).astype(bool)
    aspen[covertype == category] = True

    # Managing bad bands
    bad_bands_refl = np.zeros(426).astype(bool)
    bad_bands_refl[:8] = True
    bad_bands_refl[192:205] = True
    bad_bands_refl[284:327] = True
    bad_bands_refl[417:] = True
    refl[:, bad_bands_refl] = np.nan
    refl = refl[:, np.all(np.isnan(refl) == False, axis=0)]

    good_data = np.ones(len(xy)).astype(bool)
    #good_data[shade == 0] = False
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
    print('Fraction Training Data: {}'.format((np.sum(train) / float(len(train)))))
    print('Fraction Testing Data: {}'.format((np.sum(test) / float(len(train)))))

    #initialize weights vector and set to one for use in case where weighting == False
    weights = np.zeros(Y.shape[0])
    weights[:] = 1

    # If weighting is needed or desired, adapt code here
    if weighting == True:
        weights = np.zeros(Y.shape[0])
        needles_w = float(len(Y[train]))/float(np.sum(Y == 1))**0.8
        noneedles_w = float(len(Y[train]))/float(np.sum(Y == 0))**0.8
        print('needles_weight: {}'.format(needles_w/(needles_w+noneedles_w)))
        print('noneedles_weight: {}'.format(noneedles_w/(needles_w+noneedles_w)))
        weights[Y.flatten() == 1] = needles_w / (needles_w+noneedles_w) * 100
        weights[Y.flatten() == 0] = noneedles_w / (needles_w+noneedles_w) * 100

    # brightness normalize
    brightness = np.sqrt(np.mean(np.power(refl, 2), axis=-1))
    refl = refl / brightness[:, np.newaxis]

    # Scale brightness normalized reflectance data and save scaling information
    scaler = preprocessing.StandardScaler()
    scaler.fit(refl[train, :])
    refl = scaler.transform(refl)
    joblib.dump(scaler, 'output/trained_models/nn_aspen_scaler')

    # Return outputs of function
    return refl, Y, test, train, weights, covertype


    # Create NN model structure, and compile.  Ultimate structure desided after pretty
    # extensive (heuristically guided) testing
def nn_model(refl, num_layers, num_nodes, classes, dropout, loss_function, output_activation):
    inlayer = keras.layers.Input(shape=(refl.shape[1],))
    output_layer = inlayer

  # defining internal layer structure
    for n in range(num_layers):
        output_layer = Dense(units=num_nodes)(output_layer)

        # Activation: makes it non-linear, remove line for nested linear models. many options here.
        output_layer = keras.layers.LeakyReLU(alpha=0.3)(output_layer)

        #Regularization term: Removes nodes that are underutilized - more likely to need adjustment than leaky
        output_layer = Dropout(dropout)(output_layer)

        #Normalization: amplifys signal by normalizing finite differences to look larger. consider turning this off
        output_layer = BatchNormalization()(output_layer)

    # Activation for output layer defined here: sigmoid forces the results to look more binary. Could also be linear.
    output_layer = Dense(units=classes, activation=output_activation)(output_layer)

    # Initializing the model structure
    model = keras.models.Model(inputs=[inlayer], outputs=[output_layer])

  # Optimization function and loss functions defined here - leave as is for now
    model.compile(loss=loss_function, optimizer='adam') # change loss function to categorical_crossentropy

    return model


def nn_scenario_test(refl, Y, weights, test, train, layers_range=[4], node_range=[400], classes=2, dropout_range=[0.4],
                     loss_function='binary_crossentropy', output_activation='sigmoid', n_epochs=5, its=20):
    # reformat Y data for crossentropy loss function
    cY = np.hstack([Y, 1-Y])

    predictions = np.zeros((len(layers_range), len(dropout_range), len(node_range), len(range(its)), len(Y))).astype(np.float32)
    dim_names = ['layers', 'dropout', 'node', 'iteration']

    # Run through and train model in 5 epoch steps - basically instituting a manual stopping criteria
    # because loss is really a function of both TPR and FPRp, and we wanted to select a good combination of both.
    for _nl, nl in enumerate(layers_range):

        for _dr, dr in enumerate(dropout_range):

            for _nn, nn in enumerate(node_range):
                model = nn_model(refl, nl, nn, classes, dr, loss_function, output_activation)

                for _i in range(its):
                    print('layers {}, nodes {}, dropout {}, iteration {}'.format(nl, nn, dr, _i))
                    # weights - this is to even out the importance of small classes -
                    # needs to be updated for categorical rather than binary
                    model.fit(refl[train, ...], cY[train, ...], epochs=n_epochs, sample_weight=weights[train],
                              validation_data=(refl[test, ...], cY[test, ...], weights[test]),
                              batch_size=1000)  # can be adjusted if getting memory errors

                    pred = model.predict(refl)
                    pred = 1 - np.argmax(pred, axis=1)
                    predictions[_nl, _dr, _nn, _i, :] = pred

                    out_name = 'output/test_1.npz'
                    np.savez(out_name, predictions=predictions, dim_names=dim_names)

    return predictions, dim_names


def plotting_model_fits(Y, to_plot, covertype, layers_range, node_range, dropout_range):
    npzf = np.load('output/test_1.npz')
    predictions = npzf['predictions']
    dim_names = npzf['dim_names']

    # Predictions has dimensions: layers, dropout, nodes, iterations, predictions

    # To limit the predictions to only those that we want to plot (probably the test set) based on the 'to_plot' variable,
    # we can take a slice in the samples dimension, as:
    predictions = predictions[..., to_plot]  # this is the same as predictions[:,:,:,:,to_plot]

    # Do the same thing for the Y (truth) and for the covertype
    covertype = covertype[to_plot]

    # Now let's calculate the true positives, false_positives, etc.
    # This is how you'd do it for Y
    #true_positives  = np.sum(np.logical_and(predictions == Y, Y == 1), axis=-1)
    #false_positives = np.sum(np.logical_and(predictions != Y, predictions == 1), axis=-1)
    #true_negatives  = np.sum(np.logical_and(predictions == Y, Y == 0), axis=-1)
    #false_negatives = np.sum(np.logical_and(predictions != Y, predictions == 0), axis=-1)

    # But to be interesting, let's do it for the covertype
    un_covertype = np.unique(covertype)
    for cover in un_covertype:
        # defining the index where covertype is one of the unique covers
        cover_Y = covertype == cover

        # all these are being done over the entire 5-D array and result in 4-D arrays for each

        # defining where cover is equal to this class and where this class is predicted as aspen.
        true_positives = np.sum(np.logical_and(predictions == cover_Y, cover_Y == 1), axis=-1)

        # prediction is aspen, actual is not this cover type - only useful for aspen
        false_positives = np.sum(np.logical_and(predictions != cover_Y, predictions == 1), axis=-1)

        # prediction is not aspen, cover is this cover type
        true_negatives = np.sum(np.logical_and(predictions == cover_Y, cover_Y == 0), axis=-1)

        # prediction is not aspen, actual is not this cover type - only meaningful for aspen now
        false_negatives = np.sum(np.logical_and(predictions != cover_Y, predictions == 0), axis=-1)
        # Dimensions of the above sums are: layers, dropout, nodes, iterations

        # And some bulk numbers about the particular cover type
        num_cover = np.sum(cover_Y)
        num_not_cover = np.sum(np.logical_not(cover_Y))

        # Now lets make some aggregate plots
        axis_names = ['Layers','Dropout','Nodes']
        axis_legends = [layers_range, node_range, dropout_range]
        gs = gridspec.GridSpec(ncols=len(axis_names), nrows=2, wspace=0.1, hspace=0.4)
        fig = plt.figure(figsize=(4*len(axis_names)*1.1, 4))

        for _ax, axname in enumerate(axis_names):


            sumaxis = [0,1,2]
            sumaxis.pop(_ax)
            # Do layers TPR (axis 0)
            ax = fig.add_subplot(gs[0,_ax])
            # averaged across two of the axes - tuple is required instead of list for denoting multiple axes to sum over
            mean_slice = np.mean(true_positives / num_cover, axis=tuple(sumaxis))
            # transpose to allow different treatments to be in second axis and iterations in first for plotting
            # this is because we want the iterations on the x axis
            plt.plot(np.transpose(mean_slice))
            plt.xlabel('Iteration')
            plt.ylabel('True Positive Rate')
            plt.title(axname)
            plt.legend(axis_legends[_ax])

            # Do layers FPP (False positives / # true elements)
            ax = fig.add_subplot(gs[1,_ax])
            plt.plot(np.transpose(np.mean(false_positives / num_cover, axis=tuple(sumaxis))))
            plt.xlabel('Iteration')
            plt.ylabel('False Positive Rate Prime')
            plt.title(axname)
            plt.legend(axis_legends[_ax])

        plt.savefig(f'figs/aggregate_figs_{cover}.png', dpi=200, bbox_inches='tight')


def plotting_model_fits_singleclass(Y, to_plot, covertype, layers_range, node_range, dropout_range, matchcover='aspen'):
    npzf = np.load('output/test_1.npz')
    predictions = npzf['predictions']
    dim_names = npzf['dim_names']

    # Predictions has dimensions: layers, dropout, nodes, iterations, predictions

    # To limit the predictions to only those that we want to plot (probably the test set) based on the 'to_plot' variable,
    # we can take a slice in the samples dimension, as:
    predictions = predictions[..., to_plot]  # this is the same as predictions[:,:,:,:,to_plot]

    # Do the same thing for the Y (truth) and for the covertype
    covertype = covertype[to_plot]

    # But to be interesting, let's do it for the covertype
    un_covertype = np.unique(covertype)

    # Now lets make some aggregate plots
    axis_names = ['Layers', 'Dropout', 'Nodes']
    axis_legends = [layers_range, dropout_range, node_range]
    gs = gridspec.GridSpec(ncols=len(axis_names), nrows=len(un_covertype)+1, wspace=0.1, hspace=0.4)
    fig = plt.figure(figsize=(4 * len(axis_names)*1.2, 4 * (len(un_covertype)+1)))

    # defining where cover is equal to this class and where this class is predicted as aspen.
    true_positives = np.sum(np.logical_and(predictions == 1, covertype == matchcover), axis=-1)

    # falsely predicted 0 when this class was matchcover
    false_negatives = np.sum(np.logical_and(predictions == 0, covertype == matchcover), axis=-1)


    for _cover, cover in enumerate(un_covertype):

        # prediction is of class matchcover, actual is of multiple classes.
        false_positives = np.sum(np.logical_and(covertype == cover, predictions == 1), axis=-1)

        # And some bulk numbers about the particular cover type
        num_cover = np.sum(covertype == matchcover)
        num_not_cover = np.sum(covertype != matchcover)

        for _ax, axname in enumerate(axis_names):
            sumaxis = [0,1,2]
            sumaxis.pop(_ax)

            if cover == matchcover:
                # Do TPR
                ax = fig.add_subplot(gs[0, _ax])
                # averaged across two of the axes - tuple is required instead of list for denoting multiple axes to sum over
                mean_slice = np.mean(true_positives / num_cover, axis=tuple(sumaxis))
                # transpose to allow different treatments to be in second axis and iterations in first for plotting
                # this is because we want the iterations on the x axis
                plt.plot(np.transpose(mean_slice))
                plt.xlabel('Iteration')
                if _ax == 0:
                    plt.ylabel('True Positive Rate')
                plt.title(axname)
                plt.legend(axis_legends[_ax])
                plt.ylim([0, 1])

                # Do false negatives
                ax = fig.add_subplot(gs[1, _ax])
                # averaged across two of the axes - tuple is required instead of list for denoting multiple axes to sum over
                mean_slice = np.mean(false_negatives / num_cover, axis=tuple(sumaxis))
                # transpose to allow different treatments to be in second axis and iterations in first for plotting
                # this is because we want the iterations on the x axis
                plt.plot(np.transpose(mean_slice))
                plt.xlabel('Iteration')
                if _ax == 0:
                    plt.ylabel('False negative rate')
                plt.title(axname)
                plt.legend(axis_legends[_ax])
                plt.ylim([0, 1])

            else:
                # Do per-class false positives
                ax = fig.add_subplot(gs[1 + _cover,_ax])
                plt.plot(np.transpose(np.mean(false_positives / num_cover, axis=tuple(sumaxis))))
                plt.xlabel('Iteration')
                if _ax == 0:
                    plt.ylabel(f'Predicted {cover} but actually {matchcover}\n relative to true {matchcover}')
                plt.title(axname)
                plt.legend(axis_legends[_ax])
                plt.ylim([0,1])

    plt.savefig(f'figs/single_class_breakout_1.png', dpi=200, bbox_inches='tight')





#    true_positives = predictions[...,to_plot] - Y[np.newaxis,np.newaxis, np.newaxis, to_plot]


#    print(these_are_my_predictions.shape)



if __name__ == "__main__":
    main()
