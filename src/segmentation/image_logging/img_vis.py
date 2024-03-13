import os

from math import ceil, floor, sqrt

import numpy as np
import bokeh.palettes
from matplotlib import pyplot as plt
from matplotlib import colors

from metrics import Metric

import utils
from exception_handling import handle_exception


class ImageLogger(Metric):
    """
    Object that logs evaluated example images during training. The images may be saved to a 'Sample images' subdirectory of the main directory and to Neptune.

    Parameters:
        log_to_device: bool; whether to log the plot of the curve to the device 
        save_path: str; where to save the plots
        log_to_neptune: bool; whether to log the plot of the curve to neptune
        neptune_run: a neptune run object
        neptune_save_path: str; where to log the curve in the run
        validate: whether a validation set will be used
        thresholds: list of floats; different thresholds to be applied to prediction
        save_images_at_epoch: epoch index or list of epoch indices when images should be saved
        imgs_per_epoch: int; number of images to evaluate at active epochs
        num_epochs: number of epochs in experiment
        extension: extension of saved files
    """

    PARAMS = {
        'number of images to save': {
            'argument name': 'num_imgs',
            'default': 5
            },
        'save sample images at': {
            'argument name': 'active_epochs',
            'default': 'last'
            },
        'draw mask contour': False
        }

    def __init__(self, neptune_run = None, neptune_save_path = '',
                 validate = True, extension = 'png', exp_name = '',
                 _config_dict = None, class_names = [], *args, **kwargs):

        metric_params = _config_dict['metrics/calculation']
        self.run = neptune_run
        self.number_of_imgs = metric_params['number of images to save']
        self.to_validate = validate
        self.num_classes = metric_params.get('number_of_classes', 1)
        if self.num_classes > 1:
            self.colormap = CustomColormap(get_colors(self.num_classes))
            self.labels = class_names
            if len(self.labels) == self.num_classes:
                self.labels = self.labels[1:]
            if len(self.labels) != self.num_classes - 1:
                msg = f'Expected class labels one less or equal to the number of classes, but got {len(self.labels)} labels and {self.num_classes} classes specified.'
                raise ValueError(msg)
            self.labels = [label.replace('_', ' ') for label in self.labels]
        else:
            self.thresholds = metric_params.get_tuple('thresholds', 0.5)
            self.draw_contour = metric_params.get('draw mask contour', False)
        self.log_to_device = _config_dict['meta/technical/log to device']
        self.log_to_neptune = _config_dict['meta/technical/log to neptune']
        self.extension = extension.lower()

        self.active_epochs = metric_params['save sample images at']
        if isinstance(self.active_epochs, (str, int)):
            self.active_epochs = [self.active_epochs]
        self.active_epochs = list(self.active_epochs)
        self.do_last = 'last' in self.active_epochs
        self.REQUIRES_LAST_PASS = self.do_last

        # intialise counters
        self.epoch_idx = 1
        self.imgs_calculated = 0
        self.train = True

        # make directory
        if self.log_to_device:
            save_dir = _config_dict['meta/technical/absolute path']
            save_path =  f'{save_dir}{exp_name}/Sample images/'
            os.mkdir(save_path)
            self.save_path = save_path

        if self.run:
            self.neptune_save_path = neptune_save_path

    def calculate_batch(self, prediction, mask, x, train, last = False, *args, **kwargs):
        # note whether the epoch is a train or validation loop
        self.train = train
        if (self.epoch_idx in self.active_epochs or last and self.do_last) and self.imgs_calculated < self.number_of_imgs:
            prediction = prediction.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
            img = x.cpu().detach().numpy()

            # enumerate over every datapoint in batch
            for i, y in enumerate(mask):
                if np.any(y > 0):
                    x_, y_hat = img[i], prediction[i]
                    self.save(x_, y, y_hat, last = last)
                if self.imgs_calculated == self.number_of_imgs:
                    break
        return {}
    
    def save(self, x, y, y_hat, last = False):
        self.imgs_calculated += 1
        
        # images are reshaped to a channels-last shape
        # and pixel intensities are converted to [0, 1]
        img = np.moveaxis(x, 0, -1).squeeze()
        if np.all(img % 1 == 0): # TODO: this could be more general
            img = img / 255
        mask = y.squeeze()
        pred = y_hat.squeeze()

        # invertion needed to show grayscale image as original
        if len(img.shape) == 2:
            img = 1 - img

        # returns a matplotlib.pyplot figure with the predicitions
        if self.num_classes == 1:
            figure = visualise_binary(img, mask, pred, self.thresholds, self.draw_contour)
        else:
            figure = visualise_multiclass(img, mask, pred, self.colormap, self.labels)

        epoch_idx = self.epoch_idx - int(last)
        if self.log_to_device:
            epoch_path_name = self.save_path + 'epoch_{}/'.format(epoch_idx)
            if not os.path.isdir(epoch_path_name):
                os.mkdir(epoch_path_name)
            loop_type = 'train' if self.train else 'val'
            full_path_name = epoch_path_name + loop_type + '/'
            if not os.path.isdir(full_path_name):
                os.mkdir(full_path_name)
            plt.savefig(
                full_path_name + 'img_{}.{}'.format(self.imgs_calculated, self.extension),
                bbox_inches = 'tight'
                )
        if self.log_to_neptune:
            loop_type = 'train' if self.train else 'val'
            curr_dest = 'epoch_{}/{}/'.format(epoch_idx, loop_type)
            self.run[self.neptune_save_path + 'sample_images/' + curr_dest].log(figure)
        plt.close()

    def evaluate_batch(self, *args, **kwargs):
        return {}

    def evaluate_epoch(self, *args, **kwargs):
        if not self.train or not self.to_validate:
            # epoch index only changes after validation loop
            # or if there is no validation loop 
            self.epoch_idx += 1
        self.imgs_calculated = 0
        return {}

def threshold_pred(pred, th):
    """Helper function for visualise_preds. Returns a thresholded prediction."""

    return np.array(pred > th, dtype = float)

def configure_subplot(img, mask, label, idx, fig, n, k, draw_contour, *args, **kwargs):
    """Helper function for visualise_preds. Builds a subplot containing an image."""

    subplot = fig.add_subplot(n, k, idx)
    subplot.set_frame_on(True)
    subplot.xaxis.set_ticks_position('none')
    subplot.yaxis.set_ticks_position('none')
    subplot.xaxis.set_ticklabels([])
    subplot.yaxis.set_ticklabels([])
    subplot.xaxis.set_label_text(label, size = 16)
    plt.imshow(img, cmap = 'Greys', vmin = 0, vmax = 1)

    if draw_contour:
        # put on a contour of the ground truth mask
        plt.contour(mask, colors = 'red')

def visualise_binary(img, mask, pred, thresholds, draw_contour, *args, **kwargs):
    """
    Creates a visualisation of a model's predictions on a given image.

    Inputs:
        img: the original image
        mask: ground truth
        pred: raw (not thresholded) prediction of the model
        thresholds: list of thresholds that should be shown
        draw_contour: whether the mask should be contoured in all the images
    
    Output: a composition containing the original image, the mask, the raw prediction and the binary predictions at different thresholds, with the contour of the ground truth mask overlaid on them. Predictions and masks will be darker for larger values.
    
    NOTE: this returns a matplotlib.pyplot figure. After the function is called, the figure should be properly closed.
    """

    # calculate shape of the composition
    N = 3 + len(thresholds)
    n = floor(sqrt(N))
    k = ceil(N / n)

    if img.max() > 1:
        img = img / 255
    imgs = {'original image': img, 'ground truth': mask, 'raw prediction': pred}
    imgs.update({
        'threshold {}'.format(threshold): threshold_pred(pred, threshold) for threshold in thresholds
    })
    
    fig = plt.figure(figsize = (6 * k, 6 * n))
    for i, (name, array) in enumerate(imgs.items()):
        try:
            configure_subplot(array, mask, name, i + 1, fig, n, k, draw_contour)
        except Exception as e:
            msg = f'An exception occured while trying to visualise {name}.'
            handle_exception(e, msg)
    return fig

def get_colors(N):
    n = N - 1
    if n == 1:
        cs = ['black']
    elif n <= 10:
        cs = bokeh.palettes.Category10[n]
    elif n <= 20:
        cs = bokeh.palettes.Category20[n]
    else:
        cs = [bokeh.palettes.Turbo256[int(len(bokeh.palettes.Turbo256) / n * i)] for i in range(n)]
    return ('#ffffff', *cs)

class CustomColormap(colors.Colormap):
    
    def __init__(self, cs):
        self.colors = np.array([colors.hex2color(c) for c in cs])
        self.named_colors = cs
        self.name = 'custom_colormap'
        self.N = len(cs)
    
    def __call__(self, X, alpha = None, bytes = None):
        if np.all(X < 1):
            X = np.round(self.N * X).astype(int)
        if alpha is None:
            alpha = np.ones_like(X)
        return np.concatenate([self.colors[X].T, np.expand_dims(alpha, 0)]).T

def configure_color_subplot(img, label, idx, fig, colormap, *args, **kwargs):
    """Helper function for visualise_preds. Builds a subplot containing an image."""

    subplot = fig.add_subplot(1, 11, (3 * idx - 2, 3 * idx))
    subplot.set_frame_on(True)
    subplot.xaxis.set_ticks_position('none')
    subplot.yaxis.set_ticks_position('none')
    subplot.xaxis.set_ticklabels([])
    subplot.yaxis.set_ticklabels([])
    subplot.xaxis.set_label_text(label, size = 16)
    if idx == 1:
        plt.imshow(img, cmap = 'Greys')
    else:
        plt.imshow(colormap(img))

def add_legend(fig, cs, labels):
    # TODO: this only works for max 8 classes (excluding background) now
    # if there are more, theres should be two columns of legend
    subplot = fig.add_subplot(1, 11, (10, 11))
    subplot.set_frame_on(False)
    subplot.xaxis.set_ticks_position('none')
    subplot.yaxis.set_ticks_position('none')
    subplot.xaxis.set_ticklabels([])
    subplot.yaxis.set_ticklabels([])
    _ = [subplot.scatter([], [], marker = 's', c = c) for c in cs[1:]]
    subplot.legend(_, labels, loc = 'center', fontsize = 18, labelspacing = 1.2,
                   frameon = False, mode = 'expand', markerscale = 4.5, handletextpad = 0.25)

def visualise_multiclass(img, mask, pred, colormap, labels, *args, **kwargs): # TODO: add legend
    """
    Creates a visualisation of a model's predictions on a given image.

    Inputs:
        `img`: the original image
        `mask`: ground truth
        `pred`: raw (not thresholded) prediction of the model
    
    Output: a composition containing the original image, the mask, and the prediction.
    
    NOTE: this returns a matplotlib.pyplot figure. After the function is called, the figure should be properly closed.
    """

    imgs = {'original image': img, 'ground truth': mask, 'prediction': pred.argmax(0)}
    
    fig = plt.figure(figsize = (24, 6))
    for i, (name, array) in enumerate(imgs.items()):
        try:
            configure_color_subplot(array, name, i + 1, fig, colormap)
        except Exception as e:
            msg = f'An exception occured while trying to visualise {name}.'
            handle_exception(e, msg)
    add_legend(fig, colormap.named_colors, labels)
    return fig