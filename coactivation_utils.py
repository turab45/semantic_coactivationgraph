#import modules
import torch
import pandas as pd
import numpy as np
import networkx as nx
import torchvision
from sklearn.metrics import pairwise_distances
from copy import deepcopy
from functools import reduce
from typing import Callable, Optional, Union, List, Tuple, Dict, Hashable, Any
from collections import Counter, deque
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

## 1. Recording coactivations (activation extraction)

def export_activations(activations : torch.Tensor , init : bool , file_name : str) -> None :
    r"""Export activation from a layer and register it in `file_name` file as a `.csv`.
    Columns are neurons, lines are data.
    `init` is used to distinguish first registering, in which case header of the `.csv` is also written.
    """
    from os.path import dirname
    from os import makedirs
    
    # Create parent directory if it doesn't exist
    parent_dir = dirname(file_name)
    if parent_dir:
        makedirs(parent_dir, exist_ok=True)
    
    if len(activations.shape)>2:
        activations=activations.mean(tuple(i for i in range(2,len(activations.shape)))).cpu().numpy()
    else:
        activations=activations.cpu().numpy()
    if init:
        pd.DataFrame(activations).to_csv(file_name, mode='w', header=[f"neuron_{i}" for i in range(activations.shape[1])], index=False)
    else:
        pd.DataFrame(activations).to_csv(file_name, mode='a', header=False, index=False)

def export_ranks(folder : str, layer_name : str, extended : bool =False) -> None:
    r"""Export ranks per neuron from activations of a layer.
    Activations are looked for in `activations/` subfolder of `folder`, in `layer_name.csv`
    Ranks are saved in `ranks/` subfolder of `folder`, in `layer_name.csv`.
    If `extended` is set to `True`, also export mean rank and standard deviation of rank in `stats/` subfolder of `folder`, in `layer_name.csv`
    """
    rank_data=pd.read_csv(f"{folder}activations/{layer_name}.csv").apply(pd.to_numeric).rank()
    rank_data.to_csv(f"{folder}ranks/{layer_name}.csv", mode='w', header=list(rank_data.columns), index=False)
    if extended:
        stat_data=pd.concat([rank_data.mean().to_frame().T, rank_data.std(ddof=0).to_frame().T], ignore_index=True)
        stat_data.to_csv(f"{folder}stats/{layer_name}.csv", mode='w', header=list(rank_data.columns), index=False)

def export_all_ranks(folder : str, index_name : str="layer_index.csv", verbose : bool =True, extended : bool =True) -> None:
    r"""Export all ranks from activations of the layers.
    Layer list is retrieved from `index_name` file from `folder`. This one is read as a `.csv` and must contain a column `module name`.
    Exportation is done by calling `export_ranks`.
    """
    from os import makedirs
    makedirs(f"{folder}ranks/", exist_ok=True)
    if extended:
        makedirs(f"{folder}stats/", exist_ok=True)
    index_data=pd.read_csv(f"{folder}{index_name}")
    num_of_files=index_data.shape[0]
    if verbose:
        counter=0
    for i in range(num_of_files):
        export_ranks(folder, index_data['module name'][i], extended)
        if verbose:
            counter+=1
            print(f"{counter}/{num_of_files}")

## 2. Computing correlations

def pearson_correlation(
    A : np.ndarray,
    B : np.ndarray,
    mu_A : np.ndarray,
    mu_B : np.ndarray,
    std_A : np.ndarray,
    std_B : np.ndarray
) -> np.ndarray:
    r"""Compute pearson correlation through numpy implementation
    """
    k=A.shape[1]
    assert k==B.shape[1]
    return (A-np.atleast_2d(mu_A).T)@(B-np.atleast_2d(mu_B).T).T/(k*np.atleast_2d(std_A).T@np.atleast_2d(std_B))

def numpy_export_correlations(folder : str, index_name : str="layer_index.csv", verbose : bool =True) -> None:
    r"""Compute the correlation between layers through numpy with at most 2 files loaded at the same time and with a minimal number of file loads.
    List of layers is found in `index_name` file of `folder`.
    Data to compute correlations are taken from `ranks/` and `stats/` subfolder of folder.
    Correlations are saved in `correlations/` subfolder of `folder`, in a `layer1_name-layer2_name.csv` file.
    Lines corresponds to neurons of first layer appearing in filename.
    Columns corresponds to neurons of second layer appearing in filename
    """
    from os import makedirs
    makedirs(f"{folder}correlations/", exist_ok=True)
    index_data=pd.read_csv(f"{folder}{index_name}")
    num_of_files=index_data.shape[0]
    if verbose:
        counter=0
        total=num_of_files*(num_of_files+1)//2
    data1=pd.read_csv(f"{folder}ranks/{index_data['module name'][0]}.csv").to_numpy().T
    data2=data1
    stat1=pd.read_csv(f"{folder}stats/{index_data['module name'][0]}.csv").to_numpy()
    stat2=stat1
    if verbose:
        counter=0
    for i in range(num_of_files):
        corr=pearson_correlation(data1, data1, stat1[0], stat1[0], stat1[1], stat1[1])
        pd.DataFrame(corr).to_csv(f"{folder}correlations/{index_data['module name'][i]}-{index_data['module name'][i]}.csv", mode='w', header=False, index=False)
        if verbose:
            counter+=1
            print(f"{counter}/{total}")
        if i%2==1:
            for j in range(1,i):
                data2=pd.read_csv(f"{folder}ranks/{index_data['module name'][j]}.csv").to_numpy().T
                stat2=pd.read_csv(f"{folder}stats/{index_data['module name'][j]}.csv").to_numpy()
                corr=pearson_correlation(data1, data2, stat1[0], stat2[0], stat1[1], stat2[1])
                pd.DataFrame(corr).to_csv(f"{folder}correlations/{index_data['module name'][i]}-{index_data['module name'][j]}.csv", mode='w', header=False, index=False)
                if verbose:
                    counter+=1
                    print(f"{counter}/{total}")
            if i!=num_of_files-1:
                data2=pd.read_csv(f"{folder}ranks/{index_data['module name'][i+1]}.csv").to_numpy().T
                stat2=pd.read_csv(f"{folder}stats/{index_data['module name'][i+1]}.csv").to_numpy()
                data1, data2= data2, data1
                stat1, stat2= stat2, stat1
                corr=pearson_correlation(data1, data2, stat1[0], stat2[0], stat1[1], stat2[1])
                pd.DataFrame(corr).to_csv(f"{folder}correlations/{index_data['module name'][i+1]}-{index_data['module name'][i]}.csv", mode='w', header=False, index=False)
                if verbose:
                    counter+=1
                    print(f"{counter}/{total}")
        else:
            for j in range(i-2,-1, -1):
                data2=pd.read_csv(f"{folder}ranks/{index_data['module name'][j]}.csv").to_numpy().T
                stat2=pd.read_csv(f"{folder}stats/{index_data['module name'][j]}.csv").to_numpy()
                corr=pearson_correlation(data1, data2, stat1[0], stat2[0], stat1[1], stat2[1])
                pd.DataFrame(corr).to_csv(f"{folder}correlations/{index_data['module name'][i]}-{index_data['module name'][j]}.csv", mode='w', header=False, index=False)
                if verbose:
                    counter+=1
                    print(f"{counter}/{total}")
            if i!=num_of_files-1:
                data1=pd.read_csv(f"{folder}ranks/{index_data['module name'][i+1]}.csv").to_numpy().T
                stat1=pd.read_csv(f"{folder}stats/{index_data['module name'][i+1]}.csv").to_numpy()
                corr=pearson_correlation(data1, data2, stat1[0], stat2[0], stat1[1], stat2[1])
                pd.DataFrame(corr).to_csv(f"{folder}correlations/{index_data['module name'][i+1]}-{index_data['module name'][0]}.csv", mode='w', header=False, index=False)
                if verbose:
                    counter+=1
                    print(f"{counter}/{total}")

def export_all_correlations(folder : str , index_name : str ="layer_index.csv", verbose : bool=True) -> None:
    r"""Compute and export correlations between layers using numpy.
    """
    numpy_export_correlations(folder, index_name=index_name, verbose=verbose)

## 3. Building the coactivation graph

def build_coactivation_graph(
    folder : str,
    index_name : str = "layer_index.csv",
    thresh : float = 0.0,
    verbose : bool = True
) -> nx.DiGraph:
    r"""Build a directed graph representing coactivation relationships between neurons.
    Nodes are neurons (identified by layer and in-layer id).
    Edges are added between neurons with correlation above `thresh`.
    Edge weights are the correlation values.
    Returns a NetworkX DiGraph.
    """
    graph = nx.DiGraph()
    index_data = pd.read_csv(f"{folder}{index_name}")
    num_of_files = index_data.shape[0]
    
    if verbose:
        counter = 0
        total = num_of_files * (num_of_files + 1) // 2
    
    # Add nodes for all neurons
    for i in range(num_of_files):
        layer_name = index_data['module name'][i]
        correlations = pd.read_csv(f"{folder}correlations/{layer_name}-{layer_name}.csv", header=None).to_numpy()
        num_neurons = correlations.shape[0]
        for neuron_id in range(num_neurons):
            graph.add_node((layer_name, neuron_id), layer=layer_name, inlayer_id=neuron_id)
    
    # Add edges based on correlations within and between layers
    for i in range(num_of_files):
        layer_i = index_data['module name'][i]
        
        # Within-layer correlations
        corr_file = f"{folder}correlations/{layer_i}-{layer_i}.csv"
        correlations = pd.read_csv(corr_file, header=None).to_numpy()
        for neuron_id_1 in range(correlations.shape[0]):
            for neuron_id_2 in range(correlations.shape[1]):
                corr_val = correlations[neuron_id_1, neuron_id_2]
                if corr_val > thresh:
                    graph.add_edge((layer_i, neuron_id_1), (layer_i, neuron_id_2), weight=corr_val)
        
        if verbose:
            counter += 1
            print(f"{counter}/{total}")
        
        # Between-layer correlations
        for j in range(i):
            layer_j = index_data['module name'][j]
            corr_file = f"{folder}correlations/{layer_i}-{layer_j}.csv"
            correlations = pd.read_csv(corr_file, header=None).to_numpy()
            for neuron_id_1 in range(correlations.shape[0]):
                for neuron_id_2 in range(correlations.shape[1]):
                    corr_val = correlations[neuron_id_1, neuron_id_2]
                    if corr_val > thresh:
                        graph.add_edge((layer_i, neuron_id_1), (layer_j, neuron_id_2), weight=corr_val)
            
            if verbose:
                counter += 1
                print(f"{counter}/{total}")
    
    return graph

def save_coactivation_graph(graph : nx.DiGraph, folder : str, file_name : str = "coactivation_graph.graphml") -> None:
    r"""Save a coactivation graph to a GraphML file.
    """
    from os import makedirs
    if folder:
        makedirs(folder, exist_ok=True)
    nx.write_graphml(graph, f"{folder}{file_name}")

def load_coactivation_graph(folder : str, file_name : str = "coactivation_graph.graphml") -> nx.DiGraph:
    r"""Load a coactivation graph from a GraphML file.
    """
    return nx.read_graphml(f"{folder}{file_name}")




if __name__ == "__main__":
    # example on resnet18 on CIFAR10 with 1000 data points and all layers recorded
    # take the dataset from torchvision.datasets.CIFAR10 and the model from torchvision.models.resnet18

    # dataloader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor()), batch_size=1000, shuffle=False)
    
    # model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    # # replace ImageNet head (1000) with 10-class head
    # model.fc = torch.nn.Linear(model.fc.in_features, 10)
    # model.eval()

    # # register hooks to extract activations
    # activations = {}
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activations[name] = output.detach()
    #     return hook
    
    # for name, module in model.named_modules():
    #     module_type_name = type(module).__name__
    #     if "Conv" in module_type_name or "Linear" in module_type_name:
    #         module.register_forward_hook(get_activation(name))

    # # run the model on the dataset and export activations
    # for i, (data, target) in enumerate(dataloader):
    #     with torch.no_grad():
    #         model(data)
    #     for name in activations:
    #         export_activations(activations[name], init=(i==0), file_name=f"activations/{name}.csv")
    #     if i==0:
    #         break
    
    # # create layer index file
    # from os import makedirs
    # makedirs("activations/", exist_ok=True)
    # layer_index = pd.DataFrame({'module name': list(activations.keys())})
    # layer_index.to_csv("activations/layer_index.csv", index=False)
    
    # # export ranks
    # export_all_ranks(folder="", index_name="activations/layer_index.csv", verbose=True, extended=True)

    # # export correlations
    # export_all_correlations(folder="", index_name="activations/layer_index.csv", verbose=True)
    
    # start the neo4j server and build the coactivation graph
    graph = build_coactivation_graph(folder="", index_name="activations/layer_index.csv", thresh=0.5, verbose=True)
    save_coactivation_graph(graph, folder="", file_name="coactivation_graph.graphml")


