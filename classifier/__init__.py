import os
import numpy as np
import sys
import pickle


caffe_root = '/media/hermetico/2TB/frameworks/caffe/'
models_root = '/media/hermetico/2TB/frameworks/notebooks/TFG/dist/models/'

labels_path = os.path.join(caffe_root, 'data/daily_activities/labels.txt')
labels = list(np.loadtxt(labels_path, str, delimiter='\n'))
np_labels = np.array(labels)
google_activities_net = {
    'probs-layer' : 'probs',
    'layers' : ('pool5/7x7_s1', 'probs'),
    'deploy': os.path.join(models_root, 'finetuning-googlenet/deploy.prototxt'),
    'model': os.path.join(models_root, 'finetuning-googlenet/reference_activitiesnet.caffemodel'),
    'forest': os.path.join(models_root, 'finetuning-googlenet/forest_pool5-7x7_s1_probs.pck')
}

alexnet_activities_net = {
    'probs-layer' : 'probs',
    'layers': ('fc6', 'probs'),
    'deploy': os.path.join(models_root, 'finetuning-alexnet/deploy.prototxt'),
    'model': os.path.join(models_root, 'finetuning-alexnet/reference_activitiesnet.caffemodel'),
    'forest': os.path.join(models_root, 'finetuning-alexnet/forest_fc6_probs.pck')
}

sys.path.insert(0, caffe_root + 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()


def reshape_caffe_transformer(input_size, net):
    net.blobs['data'].reshape(input_size, 3, 227, 227)

    # Preprocessing for caffe inputs
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    return net, transformer
"""
def extract_features(net, path, layers, transformer):
    "A function which extracts the features of the an input image"
    # loads the images
    image = caffe.io.load_image(path)
    # preprocess for the images
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    net.forward()

    features = np.array([])
    for layer in layers:
        features = np.append(features, net.blobs[layer].data.copy()[0])
    return features
"""

class Classifier(object):
    def __init__(self, system):
        self.layers = system['layers']
        # loads the net
        self.net = caffe.Net(system['deploy'], system['model'], caffe.TEST)

        self.probs_layer = system['probs-layer']
        # reshapes the net and creates the transformer
        self.net, self.transformer = reshape_caffe_transformer(1, self.net)
        with open(system['forest'], 'r') as f:
            self.forest = pickle.load(f)

    def features(self, path):
        """A function which extracts the features of the an input image"""
        # loads the images
        image = caffe.io.load_image(path)
        # preprocess for the images
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        self.net.forward()

        features = np.array([])
        for layer in self.layers:
            features = np.append(features, self.net.blobs[layer].data.copy()[0])
        return features

    def probs(self, path):
        """A function which extracts the probabilities of the an input image"""
        # loads the images
        image = caffe.io.load_image(path)
        # preprocess for the images
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        self.net.forward()

        # extracts the top 5 values
        top_5_probs = np.sort(self.net.blobs[self.probs_layer].data[0].flatten())[-1:-6:-1]
        # the op 5 indexes
        top_5_indexes = self.net.blobs[self.probs_layer].data[0].flatten().argsort()[-1:-6:-1]

        top_5_probs = top_5_probs * 100 # to make a 100%
        return (top_5_indexes, top_5_probs)

    def classify_tree(self, image_path):

        features = self.features(image_path)
        prediction = self.forest.predict([features])[0]
        predicted_label = labels[prediction]
        return predicted_label

    def classify_net(self, image_path):
        top_predictions = self.probs(image_path)
        top_5_labels = np_labels[top_predictions[0]]
        return top_5_labels, top_predictions[1]


    def classify(self, path, way='complete'):
        """
        Classifies and returns the values
        :param path: the picture
        :param way: 'conmplete', 'cnn'
        :return:
        """
        if way == 'complete':
            return self.classify_tree(path)
        elif way =='cnn':
            return self.classify_net(path)


class Classifier_library(object):
    def __init__(self):

        self.google_net_system = Classifier( system = google_activities_net)
        #self.alexnet_net_system = Classifier(system = alexnet_activities_net)

        self.systems = dict(googlenet=self.google_net_system)
    def classify(self, image_path, system='googlenet', way='complete'):
        """
        Returns the label of the classifier or the top 5
        :param image_path: the picture
        :param system: 'googlenet', 'alexnet'
        :param way: 'complete', 'cnn'
        :return: always the expected outcome :)
        """
        return self.systems[system].classify(image_path, way)









