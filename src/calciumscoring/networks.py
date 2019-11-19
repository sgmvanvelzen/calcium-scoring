# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import theano
import lasagne
import numpy as np
import pickle

from os import path, makedirs
from itertools import cycle

T = theano.tensor
L = lasagne.layers


def softmax2d(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


class SingleVoxelRemover:
    """Theano-based function that removes voxels without neighbors from a binary mask"""
    def __init__(self):
        mask = T.ftensor3()
        input = mask.reshape(shape=(1, 1, mask.shape[0], mask.shape[1], mask.shape[2]))
        w = np.ones((1, 1, 3, 3, 3)).astype(theano.config.floatX)
        w[:, :, 1, 1, 1] = 0
        filtered = T.nnet.conv3d(input, theano.shared(w), (1, 1, None, None, None), (1, 1, 3, 3, 3), border_mode='half')
        output = T.clip(filtered, 0, 1).reshape(mask.shape) * mask
        self.remove_single_voxels = theano.function(inputs=[mask], outputs=output.astype('bool'), allow_input_downcast=True)

    def __call__(self, mask):
        return self.remove_single_voxels(mask)


class DilatedParallelDeeplySupervisedNetwork:
    """First stage network"""
    is3D = False
    padding = 128 + 2
    features_per_orientation = 32

    def __init__(self, config, compile_train_func=True, n_classes=7, deep_supervision=True):
        self.config = config
        self.n_classes = n_classes
        self.model_dir = path.join(
            config['scratchdir'],
            config['train_data'],
            'networks_' + str(config['train_scans']),
            'slices_' + config['model'] + '_' + config['experiment']
        )
        self.param_file = path.join(self.model_dir, 'epoch{}.pkl')
        self.network = self.compile(compile_train_func, deep_supervision)

    def compile(self, compile_train_func, deep_supervision):
        from lasagne.layers import Conv2DLayer, InputLayer

        weight_initializer = lasagne.init.HeUniform(gain='relu')
        bias_initializer = lasagne.init.Constant(0.0)
        activation_fn = lasagne.nonlinearities.elu

        # -------------------------------------------------------------------------------------------------------------------------------

        slices_var = T.ftensor4()
        normalized_slices = (T.clip(slices_var, -1000, 3000) - 130) / 1130.0

        slice_xy = (self.config['slice_size_voxels'] + self.padding) if compile_train_func else None
        channels = 3 if compile_train_func else 1
        padded_input = InputLayer((None, channels, slice_xy, slice_xy), input_var=normalized_slices)

        dilations = (2, 4, 8, 16, 32)
        n_prefilter_layers = 2
        n_filters = 32

        params = cycle([weight_initializer, bias_initializer])

        classifiers = []
        feature_extractors = []
        classifier_params = []

        for n in range(3):  # axial, sagittal, coronal
            # Define what the input is going to be
            network = padded_input
            if compile_train_func:
                network = L.SliceLayer(network, indices=n, axis=1)
                network = L.ReshapeLayer(network, shape=(-1, 1, slice_xy, slice_xy))

            # Analyze the image
            for i in range(n_prefilter_layers):
                network = Conv2DLayer(network, n_filters, (3, 3), W=next(params), b=next(params), nonlinearity=activation_fn)

            for dilation in dilations:
                network = L.DilatedConv2DLayer(network, n_filters, (3, 3), dilation, W=next(params), b=next(params), nonlinearity=activation_fn)

            network = Conv2DLayer(network, n_filters, (3, 3), W=next(params), b=next(params), nonlinearity=activation_fn)
            network = Conv2DLayer(network, self.features_per_orientation, (1, 1), W=next(params), b=next(params), nonlinearity=activation_fn)
            feature_extractors.append(network)

            network = L.dropout(network, 0.35)
            network = Conv2DLayer(network, self.n_classes, filter_size=(1, 1), W=next(params), b=next(params), nonlinearity=None)
            classifier_params.append((network.W, network.b))
            network = L.FlattenLayer(network, outdim=3)
            network = L.NonlinearityLayer(network, nonlinearity=softmax2d)
            network = L.ReshapeLayer(network, shape=(-1, self.n_classes, slices_var.shape[2] - self.padding, slices_var.shape[3] - self.padding))

            classifiers.append(network)

        self.classifiers = classifiers

        # Make classifier that uses information from all three planes
        central_voxel_features = []
        for features in feature_extractors:
            feature_vector = features
            center_xy = (self.config['slice_size_voxels'] - 1) // 2
            feature_vector = L.SliceLayer(feature_vector, indices=center_xy, axis=3)
            feature_vector = L.SliceLayer(feature_vector, indices=center_xy, axis=2)
            feature_vector = L.ReshapeLayer(feature_vector, shape=(-1, self.features_per_orientation, 1, 1))
            central_voxel_features.append(feature_vector)

        n_dense_filters = 128
        combiner_params = []
        network = L.ConcatLayer(central_voxel_features, axis=1)
        network = L.dropout(network, 0.35)
        network = Conv2DLayer(network, n_dense_filters, filter_size=(1, 1), W=next(params), b=next(params), nonlinearity=activation_fn)
        combiner_params += [network.W, network.b]
        network = L.dropout(network, 0.35)
        network = Conv2DLayer(network, self.n_classes, filter_size=(1, 1), W=next(params), b=next(params), nonlinearity=None)
        combiner_params += [network.W, network.b]
        network = L.FlattenLayer(network, outdim=2)
        network = L.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.softmax)

        combiner_input = T.ftensor4()  # concatenated feature vectors
        combiner_input_layer = L.InputLayer(shape=(None, 3 * self.features_per_orientation, None, None), input_var=combiner_input)
        combiner = Conv2DLayer(combiner_input_layer, n_dense_filters, filter_size=(1, 1), W=combiner_params[0], b=combiner_params[1], nonlinearity=activation_fn)
        combiner = Conv2DLayer(combiner, self.n_classes, filter_size=(1, 1), W=combiner_params[2], b=combiner_params[3], nonlinearity=None)
        combiner = L.FlattenLayer(combiner, outdim=3)
        combiner_softmax = L.NonlinearityLayer(combiner, nonlinearity=softmax2d)
        combiner = L.ReshapeLayer(combiner_softmax, shape=(-1, self.n_classes, combiner_input.shape[2], combiner_input.shape[3]))

        # Compile another kind of combiner that takes the orthogonal probablities into account
        ortho_combiner_probs = []
        for i, classifier in enumerate(classifiers):
            ortho_softmax = L.SliceLayer(combiner_input_layer, indices=slice(i * self.features_per_orientation, (i+1) * self.features_per_orientation), axis=1)
            ortho_softmax = Conv2DLayer(ortho_softmax, self.n_classes, filter_size=(1, 1), W=classifier_params[i][0], b=classifier_params[i][1], nonlinearity=None)
            ortho_softmax = L.FlattenLayer(ortho_softmax, outdim=3)
            ortho_softmax = L.NonlinearityLayer(ortho_softmax, nonlinearity=softmax2d)
            ortho_softmax = L.ReshapeLayer(ortho_softmax, shape=(-1, self.n_classes, combiner_input.shape[2], combiner_input.shape[3], 1))
            ortho_combiner_probs.append(ortho_softmax)

        ortho_combiner = L.ConcatLayer(ortho_combiner_probs, axis=4)
        ortho_combiner = L.ConcatLayer([
            L.FeaturePoolLayer(ortho_combiner, pool_size=len(ortho_combiner_probs), axis=4, pool_function=T.mean),
            L.ReshapeLayer(combiner_softmax, shape=(-1, self.n_classes, combiner_input.shape[2], combiner_input.shape[3], 1))
        ], axis=4)
        ortho_combiner = L.FeaturePoolLayer(ortho_combiner, pool_size=2, axis=4, pool_function=T.mean)
        ortho_combiner = L.FlattenLayer(ortho_combiner, outdim=4)

        # Make a version that ignores the final softmax but only takes the DS softmax layers into account
        ds_combiner = L.ConcatLayer(ortho_combiner_probs, axis=4)
        ds_combiner = L.FeaturePoolLayer(ds_combiner, pool_size=len(ortho_combiner_probs), axis=4, pool_function=T.mean)
        ds_combiner = L.FlattenLayer(ds_combiner, outdim=4)

        # Compile feature extraction + combination functions
        if not compile_train_func:
            self.extract_features = []
            for extractor in feature_extractors:
                self.extract_features.append(theano.function(
                    inputs=[slices_var],
                    outputs=L.get_output(extractor, deterministic=True),
                    allow_input_downcast=True
                ))

            self.classify = theano.function(
                inputs=[combiner_input],
                outputs=L.get_output(combiner, deterministic=True),
                allow_input_downcast=True
            )

            self.classify_ortho = theano.function(
                inputs=[combiner_input],
                outputs=L.get_output(ortho_combiner, deterministic=True),
                allow_input_downcast=True
            )

            self.classify_ds = theano.function(
                inputs=[combiner_input],
                outputs=L.get_output(ds_combiner, deterministic=True),
                allow_input_downcast=True
            )

        # -------------------------------------------------------------------------------------------------------------------------------

        if compile_train_func:
            half_padding = self.padding // 2
            input = L.SliceLayer(padded_input, indices=slice(half_padding, -half_padding), axis=2)
            input = L.SliceLayer(input, indices=slice(half_padding, -half_padding), axis=3)
            intensities = L.get_output(input)

            labels_var = T.wtensor4()  # N, 3, xy, xy
            learning_rate_var = T.fscalar()

            binary_labels = T.clip(labels_var, 0, 1)
            pos_prior_prob = 1.0 - T.mean(binary_labels, dtype=theano.config.floatX)

            labels = binary_labels if self.n_classes == 2 else labels_var

            # Define helper functions to quantify the classification performance
            def objective(deterministic=False):
                # Loss term of final classifier for central voxel only
                prediction = L.get_output(network, deterministic=True)
                central_labels = T.cast(labels[:, 0, (labels.shape[2] - 1) // 2, (labels.shape[3] - 1) // 2].flatten(), 'int32')
                loss = lasagne.objectives.categorical_crossentropy(prediction, central_labels).mean()

                overall_loss = loss

                if deep_supervision:
                    # Loss terms of orthogonal planes
                    for n, classifier in enumerate(classifiers):
                        prediction = L.get_output(classifier, deterministic=deterministic)
                        prediction_flattened = prediction.dimshuffle(1, 0, 2, 3).flatten(2).dimshuffle(1, 0)
                        labels_flattened = T.cast(labels[:, n, :, :].flatten(), 'int32')
                        loss = lasagne.objectives.categorical_crossentropy(prediction_flattened, labels_flattened)

                        #sample_weights = T.cast(1.0 / (1.0 + T.exp(-5.0 * intensities[:, n, :, :].flatten())), theano.config.floatX)
                        #weighted_loss = loss * sample_weights
                        #loss = T.sum(weighted_loss) / T.sum(sample_weights)
                        loss = loss.mean()

                        overall_loss += 0.05 * loss

                # L2 penalty
                params = []
                for classifier in classifiers:
                    params += L.get_all_params(classifier, regularizable=True)
                params += L.get_all_params(network, regularizable=True)
                params = lasagne.utils.unique(params)
                overall_loss += 0.00005 * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)

                return overall_loss

            def accuracy(binary=True, deterministic=False):
                prediction = L.get_output(network, deterministic=deterministic)
                predicted_labels = T.argmax(prediction, axis=1)
                if binary:
                    predicted_labels = T.clip(predicted_labels, 0, 1)
                true_labels = binary_labels if binary else labels
                central_labels = T.cast(true_labels[:, 0, (true_labels.shape[2] - 1) // 2, (true_labels.shape[3] - 1) // 2].flatten(), 'int32')
                return T.mean(T.eq(predicted_labels, central_labels), dtype=theano.config.floatX)

            # Compile the training functions
            loss = objective()

            params = []
            if deep_supervision:
                for classifier in classifiers:
                    params += L.get_all_params(classifier, trainable=True)
            params += L.get_all_params(network, trainable=True)
            params = lasagne.utils.unique(params)
            updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate_var)

            learning_rate_param = theano.In(variable=learning_rate_var, value=self.config['lr'])
            self.train = theano.function(
                inputs=[slices_var, labels_var, learning_rate_param],
                outputs=[loss, accuracy(binary=True), accuracy(binary=False), pos_prior_prob],
                updates=updates,
                allow_input_downcast=True
            )

            # Compile validation function
            self.validate = theano.function(
                inputs=[slices_var, labels_var],
                outputs=[objective(deterministic=True), accuracy(binary=True, deterministic=True), accuracy(binary=False, deterministic=True), pos_prior_prob],
                allow_input_downcast=True
            )

        return network

    def count_params(self):
        return L.count_params(self.network)  # does not include the additional softmax layers

    def save(self, epoch):
        filename = self.param_file.format(epoch)

        dirname = path.dirname(filename)
        if not path.exists(dirname):
            makedirs(dirname)

        with open(filename, 'wb') as f:
            for classifier in self.classifiers:
                pickle.dump(L.get_all_param_values(classifier), f, -1)
            pickle.dump(L.get_all_param_values(self.network), f, -1)

    def restore(self, epoch, raise_errors=True):
        filename = self.param_file.format(epoch)
        return self.restore_from_file(filename, raise_errors)

    def restore_from_file(self, filename, raise_errors=True):
        if not path.exists(filename):
            if raise_errors:
                raise IOError('Could not restore network from disk: file does not exist')
            else:
                return False

        with open(filename, 'rb') as f:
            try:
                for classifier in self.classifiers:
                    L.set_all_param_values(classifier, pickle.load(f))
                L.set_all_param_values(self.network, pickle.load(f))
            except UnicodeDecodeError:
                for classifier in self.classifiers:
                    L.set_all_param_values(classifier, pickle.load(f, encoding='latin1'))
                L.set_all_param_values(self.network, pickle.load(f, encoding='latin1'))

        return True


class UndilatedConvNet:
    """Second stage network"""
    is3D = False

    def __init__(self, config, compile_train_func=True):
        self.config = config
        self.model_dir = path.join(
            config['scratchdir'],
            config['train_data'],
            'networks_' + str(config['train_scans']),
            'voxels_' + config['model'] + '_' + config['experiment']
        )
        self.param_file = path.join(self.model_dir, 'epoch{}.pkl')
        self.network = self.compile(compile_train_func)

    def compile(self, compile_train_func):
        from lasagne.layers import Conv2DLayer, MaxPool2DLayer

        n_classes = int(self.config['model'][0])

        # Construct the network
        weight_initializer = lasagne.init.Orthogonal()
        activation_fn = lasagne.nonlinearities.elu

        patch_vars = [T.ftensor4(), T.ftensor4(), T.ftensor4()]
        label_var = T.ivector()
        learning_rate_var = T.fscalar()

        concurrent_networks = []
        for i in range(3):
            if len(concurrent_networks) == 0:
                weights = 5 * [weight_initializer]
            else:
                weights = []
                for layer in L.get_all_layers(concurrent_networks[0]):
                    try:
                        weights.append(layer.W)
                    except AttributeError:
                        continue

            input_shape = (None, 1, self.config['patch_size_voxels'], self.config['patch_size_voxels'])
            network = L.InputLayer(input_shape, input_var=patch_vars[i])

            network = Conv2DLayer(network, 24, (5, 5), W=weights[0], nonlinearity=activation_fn)
            network = L.batch_norm(network)

            network = MaxPool2DLayer(network, (2, 2))
            network = Conv2DLayer(network, 32, (3, 3), W=weights[1], nonlinearity=activation_fn)
            network = L.batch_norm(network)

            network = MaxPool2DLayer(network, (2, 2))
            network = Conv2DLayer(network, 48, (3, 3), W=weights[2], nonlinearity=activation_fn)
            network = L.batch_norm(network)

            network = MaxPool2DLayer(network, (2, 2))
            network = Conv2DLayer(network, 32, (1, 1), W=weights[3], nonlinearity=activation_fn)
            network = L.batch_norm(network)

            network = L.FlattenLayer(network)
            concurrent_networks.append(network)

        network = L.ConcatLayer(concurrent_networks)

        network = L.DropoutLayer(network, p=0.5)
        network = L.DenseLayer(network, 256, W=weight_initializer)
        network = L.batch_norm(network)

        network = L.DropoutLayer(network, p=0.5)
        network = L.DenseLayer(network, n_classes, nonlinearity=lasagne.nonlinearities.softmax)

        # Compile the network
        deterministic_prediction = L.get_output(network, deterministic=True)
        self.classify = theano.function(inputs=patch_vars, outputs=deterministic_prediction, allow_input_downcast=True)

        if compile_train_func:
            true_label = T.clip(label_var, 0, n_classes - 1)

            # Define helper functions to quantify the classification performance
            def objective(prediction):
                logloss = lasagne.objectives.categorical_crossentropy(prediction, true_label).mean()
                params = L.get_all_params(network, regularizable=True)
                l2_penalty = 0.00001 * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
                return logloss + l2_penalty

            def accuracy(prediction):
                predicted_label = T.argmax(prediction, axis=1)
                return T.mean(T.eq(predicted_label, true_label), dtype=theano.config.floatX)

            # Compile the training function
            noisy_prediction = L.get_output(network)
            loss = objective(noisy_prediction)

            params = L.get_all_params(network, trainable=True)
            updates = lasagne.updates.adam(loss, params, learning_rate_var)

            learning_rate_param = theano.In(variable=learning_rate_var, value=self.config['lr'])

            self.train = theano.function(
                inputs=patch_vars + [label_var, learning_rate_param],
                outputs=[loss, accuracy(noisy_prediction)],
                updates=updates,
                allow_input_downcast=True
            )

            # Compile validation function
            self.validate = theano.function(
                inputs=patch_vars + [label_var],
                outputs=[objective(deterministic_prediction), accuracy(deterministic_prediction)],
                allow_input_downcast=True
            )

        return network

    def count_params(self):
        return L.count_params(self.network)

    def save(self, epoch):
        filename = self.param_file.format(epoch)

        dirname = path.dirname(filename)
        if not path.exists(dirname):
            makedirs(dirname)

        with open(filename, 'wb') as f:
            pickle.dump(L.get_all_param_values(self.network), f, -1)

    def restore(self, epoch, raise_errors=True):
        filename = self.param_file.format(epoch)

        if not path.exists(filename):
            if raise_errors:
                raise IOError('Could not restore network from disk: file does not exist')
            else:
                return False

        with open(filename, 'rb') as f:
            try:
                L.set_all_param_values(self.network, pickle.load(f))
            except UnicodeDecodeError:
                L.set_all_param_values(self.network, pickle.load(f, encoding='latin1'))

        return True
