

import numpy as np
import pandas as pd
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from tqdm import tqdm
import joblib
#from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D, Concatenate, SeparableConv2D, Dropout
from itertools import product



mpl.use('Agg')


def main():
    file_path = '~/Google Drive File Stream/My Drive/CB_share/NEON/cover_classification/extraction_output/cover_extraction_20201021.csv'
    #file_path = 'data/cover_extraction_20201021.csv'
    layers_range = [2]
    node_range = [250, 550]
    dropout_range = [0.4]

    data_munge_dir = 'munged/data_splits.npz' # (multiclass test 6)
    data_munge_dir = 'munged/data_splits_7.npz' # (multiclass test 7)

    if os.path.isfile(data_munge_dir):
        npzf = np.load(data_munge_dir, allow_pickle=True)
        refl = npzf['refl']
        Y = npzf['Y']
        test = npzf['test']
        train = npzf['train']
        weights = npzf['weights']
        covertype = npzf['covertype']
        y_labels = npzf['y_labels']
    else:
        refl, Y, test, train, weights, covertype, y_labels = manage_datasets(file_path, category=None)
        np.savez(data_munge_dir, refl=refl, Y=Y, test=test, train=train, weights=weights, covertype=covertype, y_labels=y_labels)

    output_filename = 'output/multiclass_test_5.npz' #basic weighting
    output_filename = 'output/multiclass_test_6.npz' #shade masked, weighted
    output_filename = 'output/multiclass_test_7.npz' #shade masked, weighted, 1000 x 1000 grids

    preds, dim_names = nn_scenario_test(output_filename, refl, Y, weights, test, train, y_labels, covertype, n_epochs=2, its=50,
                                        layers_range=layers_range, node_range=node_range, dropout_range=dropout_range, classes=Y.shape[1])

    plotting_model_fits(Y, test, covertype, layers_range, node_range, dropout_range, y_labels)
    for cover in y_labels:
        plotting_model_fits_singleclass(output_filename, test, covertype, layers_range, node_range, dropout_range,
                                        matchcover=cover, classlabels=y_labels)

    plot_confusion_matrix(output_filename, train, np.argmax(Y,axis=1), layers_range, node_range, dropout_range, y_labels)



def manage_datasets(import_data, category='aspen', weighting=False):
    # Import extraction csv
    data_set = pd.read_csv(import_data)

    # extract x y coordinates for each pixel
    xy = np.array(data_set[['X_UTM', 'Y_UTM']])
    shade = np.array(data_set['hade_B_1']).flatten()
    #tch = np.array(data_set['_tch_B_1']).flatten()

    # extract reflectance data from csv
    refl = np.array(data_set)[:, -427:-1].astype(np.float32)

    # Extract cover type data
    covertype = np.array(data_set[['covertype']]).flatten()
    print(np.unique(covertype))

    # Managing bad bands
    bad_bands_refl = np.zeros(426).astype(bool)
    bad_bands_refl[:8] = True
    bad_bands_refl[192:205] = True
    bad_bands_refl[284:327] = True
    bad_bands_refl[417:] = True
    refl[:, bad_bands_refl] = np.nan
    refl = refl[:, np.all(np.isnan(refl) == False, axis=0)]

    good_data = np.ones(len(xy)).astype(bool)
    good_data[shade == 0] = False
    good_data = good_data.flatten()
    # if there are any nans in the entire row, remove that pixel
    good_data[np.any(np.isnan(refl), axis=1)] = False
    good_data[np.all(refl == 0, axis=1)] = False

    refl = refl[good_data, ...]
    covertype = covertype[good_data, ...]
    xy = xy[good_data, :]

    if category is not None:
        Y = np.zeros((len(covertype),2)).astype(bool)
        Y[:,1] = covertype == category
        Y[:,0] = covertype != category
        y_labels = ['not ' + category, category]
    else:
        # If None, one-hot-encode
        unique_categories = np.unique(covertype)
        Y = np.zeros((len(covertype), len(unique_categories))).astype(bool)
        for _cat, cat in enumerate(unique_categories):
            Y[:,_cat] = covertype == cat
        y_labels = list(unique_categories)


    np.random.seed(13)
    perm = np.random.permutation(Y.shape[0])
    train = np.zeros(perm.shape).astype(bool)

    n_xstep = 1000
    n_ystep = 1000
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


    for cover in np.unique(covertype):
        fraction = 100*np.sum(covertype[train] == cover) / len(covertype[train])
        print(f'% Training data is {cover}: {fraction}')
    for cover in np.unique(covertype):
        fraction = 100*np.sum(covertype[test] == cover) / len(covertype[test])
        print(f'% Testing data is {cover}: {fraction}')

    #initialize weights vector and set to one for use in case where weighting == False
    weights = np.ones(Y.shape[0])


    standardized_count = len(covertype[train]) / float(len(y_labels))
    for _cover, cover in enumerate(y_labels):
        w = standardized_count / np.sum(covertype[train] == cover) 
        weights[cover == covertype] = w
        print(f'cover: {cover}, weight: {w}')

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
    #scaler = preprocessing.StandardScaler()
    #scaler.fit(refl[train, :])
    #refl = scaler.transform(refl)
    #joblib.dump(scaler, 'output/trained_models/nn_aspen_scaler')

    # Return outputs of function
    return refl, Y, test, train, weights, covertype, y_labels


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


def nn_scenario_test(output_filename, refl, Y, weights, test, train, y_labels, covertype, layers_range=[4], node_range=[400], classes=2, dropout_range=[0.4],
                     loss_function='categorical_crossentropy', output_activation='sigmoid', n_epochs=5, its=20):

    predictions = np.zeros((len(layers_range), len(dropout_range), len(node_range), len(range(its)), len(Y))).astype(np.float32)
    predictions[...] = np.nan
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
                    model.fit(refl[train, ...], Y[train, ...], epochs=n_epochs, sample_weight=weights[train],
                              validation_data=(refl[test, ...], Y[test, ...], weights[test]),
                              batch_size=1000)  # can be adjusted if getting memory errors

                    pred = model.predict(refl)
                    pred = np.argmax(pred, axis=1)
                    predictions[_nl, _dr, _nn, _i, :] = pred

                    np.savez(output_filename, predictions=predictions, dim_names=dim_names)

                for cover in y_labels:
                    plotting_model_fits_singleclass(output_filename, test, covertype, layers_range, node_range, dropout_range,
                                                    matchcover=cover, classlabels=y_labels)

    return predictions, dim_names


def plotting_model_fits(output_filename, to_plot, covertype, layers_range, node_range, dropout_range, y_labels):
    npzf = np.load(output_filename)
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
    #un_covertype = np.unique(covertype)
    for _c, cover in enumerate(y_labels):
        # defining the index where covertype is one of the unique covers
        cover_Y = covertype == cover

        # all these are being done over the entire 5-D array and result in 4-D arrays for each

        # defining where cover is equal to this class and where this class is predicted as aspen.
        #### NOT SURE ABOUT INDEXING HERE NOW THAT IT"S MULTICLASS
        true_positives = np.nansum(np.logical_and(predictions == cover_Y, cover_Y == _c), axis=-1)

        # prediction is aspen, actual is not this cover type - only useful for aspen
        false_positives = np.nansum(np.logical_and(predictions != cover_Y, predictions == 1), axis=-1)

        # prediction is not aspen, cover is this cover type
        true_negatives = np.nansum(np.logical_and(predictions == cover_Y, cover_Y == 0), axis=-1)

        # prediction is not aspen, actual is not this cover type - only meaningful for aspen now
        false_negatives = np.nansum(np.logical_and(predictions != cover_Y, predictions == 0), axis=-1)
        # Dimensions of the above sums are: layers, dropout, nodes, iterations

        # And some bulk numbers about the particular cover type
        num_cover = np.nansum(cover_Y)
        num_not_cover = np.nansum(np.logical_not(cover_Y))

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
            mean_slice = np.nanmean(true_positives / num_cover, axis=tuple(sumaxis))
            # transpose to allow different treatments to be in second axis and iterations in first for plotting
            # this is because we want the iterations on the x axis
            plt.plot(np.transpose(mean_slice))
            plt.xlabel('Iteration')
            plt.ylabel('True Positive Rate')
            plt.title(axname)
            plt.legend(axis_legends[_ax])

            # Do layers FPP (False positives / # true elements)
            ax = fig.add_subplot(gs[1,_ax])
            plt.plot(np.transpose(np.nanmean(false_positives / num_cover, axis=tuple(sumaxis))))
            plt.xlabel('Iteration')
            plt.ylabel('False Positive Rate Prime')
            plt.title(axname)
            plt.legend(axis_legends[_ax])

        baseout = os.path.splitext(os.path.basename(output_filename))[0]
        plt.savefig(f'figs/{baseout}_{cover}.png', dpi=200, bbox_inches='tight')


def plotting_model_fits_singleclass(output_filename, to_plot, covertype, layers_range, node_range, dropout_range, matchcover, classlabels):
    npzf = np.load(output_filename)
    predictions = npzf['predictions']
    dim_names = npzf['dim_names']

    # Predictions has dimensions: layers, dropout, nodes, iterations, predictions

    # To limit the predictions to only those that we want to plot (probably the test set) based on the 'to_plot' variable,
    # we can take a slice in the samples dimension, as:
    predictions = predictions[..., to_plot]  # this is the same as predictions[:,:,:,:,to_plot]

    # Do the same thing for the Y (truth) and for the covertype
    covertype = covertype[to_plot]

    # But to be interesting, let's do it for the covertype

    # Now lets make some aggregate plots
    axis_names = ['Layers', 'Dropout', 'Nodes']
    axis_legends = [layers_range, dropout_range, node_range]
    fig = plt.figure(figsize=(4 * len(axis_names)*1.2, 4 * (len(classlabels)+1)))
    gs = gridspec.GridSpec(ncols=len(axis_names), nrows=len(classlabels)+1, wspace=0.1, hspace=0.4)
    # defining where cover is equal to this class and where this class is predicted as aspen.
    true_positives = np.nansum(np.logical_and(predictions == classlabels.index(matchcover), covertype == matchcover), axis=-1)

    # falsely predicted 0 when this class was matchcover
    false_negatives = np.nansum(np.logical_and(predictions != classlabels.index(matchcover), covertype == matchcover), axis=-1)


    for _cover, cover in enumerate(classlabels):

        # prediction is of class matchcover, actual is of multiple classes.
        false_positives = np.nansum(np.logical_and(covertype == cover, predictions == classlabels.index(matchcover)), axis=-1)

        # And some bulk numbers about the particular cover type
        num_cover = np.nansum(covertype == matchcover)
        num_not_cover = np.nansum(covertype != matchcover)

        for _ax, axname in enumerate(axis_names):
            sumaxis = [0,1,2]
            sumaxis.pop(_ax)

            if cover == matchcover:
                # Do TPR
                ax = fig.add_subplot(gs[0, _ax])
                # averaged across two of the axes - tuple is required instead of list for denoting multiple axes to sum over
                mean_slice = np.nanmean(true_positives / num_cover, axis=tuple(sumaxis))
                # transpose to allow different treatments to be in second axis and iterations in first for plotting
                # this is because we want the iterations on the x axis
                plt.plot(np.transpose(mean_slice))
                plt.xlabel('Iteration')
                if _ax == 0:
                    plt.ylabel('True Positive Rate')
                plt.title(axname)
                plt.legend(axis_legends[_ax])
                plt.ylim([-0.05, 1.05])

                # Do false negatives
                ax = fig.add_subplot(gs[1, _ax])
                # averaged across two of the axes - tuple is required instead of list for denoting multiple axes to sum over
                mean_slice = np.nanmean(false_negatives / num_cover, axis=tuple(sumaxis))

                # transpose to allow different treatments to be in second axis and iterations in first for plotting
                # this is because we want the iterations on the x axis
                plt.plot(np.transpose(mean_slice))
                plt.xlabel('Iteration')
                if _ax == 0:
                    plt.ylabel('False negative rate')
                plt.title(axname)
                plt.legend(axis_legends[_ax])
                plt.ylim([-0.05, 1.05])

            else:
                # Do per-class false positives
                offset = 0
                if classlabels.index(matchcover) < _cover:
                    offset = -1
                ax = fig.add_subplot(gs[2 + _cover + offset,_ax])
                plt.plot(np.transpose(np.nanmean(false_positives / num_cover, axis=tuple(sumaxis))))
                plt.xlabel('Iteration')
                if _ax == 0:
                    plt.ylabel(f'Predicted {cover} but actually {matchcover}\n relative to true {matchcover}')
                plt.title(axname)
                plt.legend(axis_legends[_ax])
                plt.ylim([-0.05, 1.05])

    baseout = os.path.splitext(os.path.basename(output_filename))[0]
    plotname = f'figs/multiclass/{baseout}_{matchcover}_class_breakout_5.png'
    print(f'Saving {plotname}')
    plt.savefig(plotname, dpi=200, bbox_inches='tight')
    plt.clf()
    del fig


def add_cm_plot(cm, ax, display_labels=None, cmap='viridis', include_values=True, values_format=None, xticks_rotation='vertical'):
    # Borrowed in large part from https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/metrics/_plot/confusion_matrix.py#L135
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    text_ = None
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0

        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min

            if values_format is None:
                text_cm = format(cm[i, j], '.2g')
                if cm.dtype.kind != 'f':
                    text_d = format(cm[i, j], 'd')
                    if len(text_d) < len(text_cm):
                        text_cm = text_d
            else:
                text_cm = format(cm[i, j], values_format)

            text_[i, j] = ax.text(
                j, i, text_cm,
                ha="center", va="center",
                color=color)

    if display_labels is None:
        display_labels = np.arange(n_classes)
    else:
        display_labels = display_labels

    plt.colorbar(im_, ax=ax)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels,
           ylabel="True label",
           xlabel="Predicted label")

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)


def plot_confusion_matrix(output_filename, train, covertype, layers_range, node_range, dropout_range, classlabels):
    npzf = np.load(output_filename)
    predictions = npzf['predictions']
    dim_names = npzf['dim_names']


    for _nl, nl in enumerate(layers_range):

        for _dr, dr in enumerate(dropout_range):

            for _nn, nn in enumerate(node_range):
                fig = plt.figure(figsize=(13,4))
                gs = gridspec.GridSpec(ncols=2, nrows=1, wspace=0.1, hspace=0.6)

                ax = fig.add_subplot(gs[0,0])

                pred = predictions[_nl, _dr, _nn, -1, :]
                cm = confusion_matrix(covertype[train], pred[train])

                add_cm_plot(cm, ax, classlabels)
                plt.title('Train') 



                ax = fig.add_subplot(gs[0,1])

                cm = confusion_matrix(covertype[np.logical_not(train)], pred[np.logical_not(train)])

                add_cm_plot(cm, ax, classlabels)
                plt.title('Test') 


                baseout = os.path.splitext(os.path.basename(output_filename))[0]
                plotname = f'figs/cm/{baseout}_nl_{nl}_dr_{dr}_nn_{nn}_class_breakout_5.png'
                print(f'Saving {plotname}')
                plt.savefig(plotname, dpi=200, bbox_inches='tight')
                plt.clf()
                del fig






#    true_positives = predictions[...,to_plot] - Y[np.newaxis,np.newaxis, np.newaxis, to_plot]


#    print(these_are_my_predictions.shape)



if __name__ == "__main__":
    main()
