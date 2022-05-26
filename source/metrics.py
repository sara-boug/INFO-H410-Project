import tensorflow.keras.backend as K

"""
 The metrics implemented below have been inspired from : 
     - https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
"""


def multi_class_dice_score(ground_truth, prediction, per_class=False):
    def _dice_index(ground_truth, prediction, smooth=1e-6):
        # flatten label and prediction tensors
        targets = K.flatten(ground_truth)
        inputs = K.flatten(prediction)
        intersection = K.sum(targets * inputs)
        dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
        return 1 - dice

    index = 0
    dice_indices = []
    for i in range(0, ground_truth.shape[-1]):
        dice = _dice_index(ground_truth[..., i], prediction[..., i])
        dice_indices.append(dice)
        index += dice
    if per_class:
        return dice_indices
    else:
        return index / 5


def multi_class_jaccard_score(ground_truth, prediction, per_class=False):
    def _jaccard_index(ground_truth, prediction, smooth=1e-6):
        inputs = K.flatten(prediction)
        targets = K.flatten(ground_truth)
        intersection = K.sum(targets * inputs)
        total = K.sum(targets) + K.sum(inputs)
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)
        return 1 - IoU

    index = 0
    jaccard_indices = []
    for i in range(0, ground_truth.shape[-1]):
        jaccard = _jaccard_index(ground_truth[..., i], prediction[..., i])
        jaccard_indices.append(jaccard)
        index += jaccard

    if per_class:
        return jaccard_indices
    else:
        return index / 5
