from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn


def cluster_conv(conv, clastering_method, n_clusters):
    """
    Cluster filters of nn.Con2d layer and replace them with centroids. Inplace.
    
    Parameters
    ----------
    conv: nn.Con2d instance
    
    clustering_method: clustering method instance
    
    n_clusters: int
    
    Returns
    -------
    n_unique_filters: number of unique filters in the modified layer
    """
    convs = conv.weight.data # out_channels x in_channels x kernel x kernel
    vectors = convs.flatten(start_dim=1).numpy() # out_channels x (in_channels * kernel * kernel)

    assert 1 <= n_clusters <= convs.size(0)
    
    clastering_method.set_params(n_clusters=n_clusters)
    clastering_method.fit(vectors)
    labels = clastering_method.labels_ # cluster numbers of each filter

    label_to_cenroid = dict()

    for label in set(labels):
        cluster = convs[np.flatnonzero(labels == label)]
        centroid = torch.mean(cluster, axis=0)
        label_to_cenroid[label] = centroid
    
    weight_pruned = torch.stack([label_to_cenroid[label] for label in labels])
    conv.weight.data = weight_pruned

    n_unique_filters = len(set(labels))

    return n_unique_filters


def prune(model, clastering_method, cluster_nums, copy_=True):
    """
    Prune ResNet-20 model by clustering filters of convolutional layers and 
    replacing them with centroids.
    
    Parameters
    ----------
    model: ResNet-20 model instance
    
    clustering_method: clustering method instance
    
    cluster_nums: sequence of length 4, n_clusters for each ResNet-20 layer
    
    copy_: if copy the model or prune inplace
    
    Returns
    -------
    model: pruned model on device=='cpu'
    n_unique_conv_parameters: int, number of unique parameters from all conv layers after pruning
    """
    assert len(cluster_nums) == 4

    n_unique_conv_parameters = 0

    model.to('cpu')

    if copy_:
	    model = deepcopy(model)

    layers = model.conv1, model.layer1, model.layer2, model.layer3

    for n_clusters, layer in zip(cluster_nums, layers):
        for module_name, module in layer.named_modules():
            if isinstance(module, nn.Conv2d):
                n_unique_filters = cluster_conv(module, clastering_method, n_clusters)
                one_filter_parameters_num = module.weight.numel() / module.out_channels
                n_unique_conv_parameters += n_unique_filters * one_filter_parameters_num
    
    return model, int(n_unique_conv_parameters)
