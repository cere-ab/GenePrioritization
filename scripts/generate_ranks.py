from pathlib import Path
import json
import pandas as pd
from time import perf_counter
import networkx as nx
from tqdm import tqdm
#import argparse
from utils.dataclass import getData
from infer import InferenceEngine
from model.model import GNN7L_SAGEConv
import torch
import torch_geometric
#from torch_geometric.nn import GNNExplainer
from torch_geometric.explain import Explainer, GNNExplainer
from utils.helpers import time_keeper
from lib.GraphSVX.src.explainers import GraphSVX
import random
import gc


def predict_candidate_genes_gnn_explainer_only2(
        model, data, prediction_logits,
        device, explanation_nodes_ratio=1,
        masks_for_seed=10, G=None, num_pos='all'
    ):
    """
    """
    x           = data.x
    labels      = data.y
    edge_index  = data.edge_index

    ranking         = {}
    candidates      = {}
    nodes_with_idxs = {}
    subg_numnodes_d = {}

    nodes_names = list(G.nodes)

    # Take all positive genes
    i = 0
    for node in G:
        if labels[i] == 0:
            nodes_with_idxs[node] = i
        i += 1
    
    # print('[+]', len(nodes_with_idxs), 'positive nodes found in the graph')
    print("{} positive labelled nodes found in data graph (PPI)".format(
        len(nodes_with_idxs)
    ))

    if num_pos == "all":
        num_pos = len(nodes_with_idxs)

    # Get the subgraphs of every positive nodes
    for node in tqdm(nodes_with_idxs, desc='Identifying subgraphs of +ve nodes'):
        idx = nodes_with_idxs[node]

        subg_nodes, subg_edge_index, subg_mapping, subg_edge_mask = \
            torch_geometric.utils.k_hop_subgraph(idx, 1, edge_index)
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]
    
    del subg_nodes, subg_edge_index, subg_mapping, subg_edge_mask
    nodes_explained = 0
    # Get explanations of all the positive genes
    for node in tqdm(nodes_with_idxs, desc='Explaining subgraphs of +ve nodes'):
        idx = nodes_with_idxs[node]

        candidates[node] = {}

        mean_mask = torch.zeros(edge_index.shape[1]).to(device)

        for i in range(masks_for_seed):
            # explainer = GNNExplainer(
            #     model,
            #     epochs=200,
            #     return_type='log_prob',
            #     num_hops=1,
            #     log=False
            # )
            # node_feat_mask, edge_mask = explainer.explain_node(idx, x, edge_index)
            
            algoObj=GNNExplainer(
                epochs=100,
                lr=0.01
            )
            
            explainerObj = Explainer(
                model=model,
                algorithm=algoObj,
                explanation_type='model',
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=dict(
                    mode='multiclass_classification',
                    task_level='node',
                    return_type='log_probs'
                ),
            )
            explanation = explainerObj(
                x=x,
                edge_index=edge_index,
                index=idx
            )
            
            mean_mask += explanation.edge_mask.to(device)
            del explainerObj, algoObj


        mean_mask = torch.div(mean_mask, masks_for_seed)

        # values, indices = torch.topk(mean_mask, subg_numnodes_d[idx][1]) #take ordered list of all edges
        mean_mask = torch.div(mean_mask, masks_for_seed)

        num_nodes = int(round(subg_numnodes_d[idx][0]*explanation_nodes_ratio))

        threshold = torch.mean(mean_mask) #to discuss when an edge in important or not
        hard_mean_mask = (mean_mask >= threshold).to(torch.float) #>=

        values = mean_mask[hard_mean_mask == 1] ###check if correct!!!!
        indices = hard_mean_mask.nonzero()

        del mean_mask, hard_mean_mask, threshold
        gc.collect()

        seen_genes = set()

        for i in range(len(indices)):
            src = edge_index[0][indices[i]]
            trgt = edge_index[1][indices[i]]

            src_name = nodes_names[src]
            trgt_name = nodes_names[trgt]

            src_pred = prediction_logits[src]
            trgt_pred = prediction_logits[trgt]

            # if gene has not been seen and it is not the explained node
            # we add it to the seen genes set
            if src_name != node:
                seen_genes.add(src_name)
            if trgt_name != node:
                seen_genes.add(trgt_name)

            if torch.argmax(src_pred) == 1: # LP # no needed here but unlabelled
                if src_name not in candidates[node]:
                    candidates[node][src_name] = values[i].item()
                else:
                    candidates[node][src_name] += values[i].item()

            if torch.argmax(trgt_pred) == 1: # LP #no needed here but unlabelled
                if trgt_name not in candidates[node]:
                    candidates[node][trgt_name] = values[i].item()
                else:
                    candidates[node][trgt_name] += values[i].item()
            
            # when the seen geens set reaches the num_nodes threshold
            # break the loop
            if len(seen_genes) >= num_nodes:
                break
        
        nodes_explained += 1
        if num_pos != len(nodes_with_idxs) and nodes_explained >= num_pos:
            break

    for seed in tqdm(candidates, desc="Scoring the candidate genes"):
        for candidate in candidates[seed]:
            if candidate not in ranking:
                ranking[candidate] = [1, candidates[seed][candidate]]#.item()]
            else:
                ranking[candidate][0] += 1
                ranking[candidate][1] += candidates[seed][candidate]#.item()

    print("Now sorting the genes as per ranking...", end='')
    # sorted_ranking = sorted(
    #     ranking, 
    #     key=lambda x: (
    #         ranking[x][0],
    #         ranking[x][1]
    #     ),
    #     reverse=True)
    rankdf = pd.DataFrame.from_dict(
        ranking,
        orient="index",
        columns=["Pseudo-labels", "Score"] #BUG: These column headers could be wrong
    )
    rankdf["GeneID"] = rankdf.index
    rankdf.reset_index(drop=True, inplace=True)
    rankdf = rankdf[["GeneID", "Score", "Pseudo-labels"]]

    rankdf.sort_values(
        ["Score", "Pseudo-labels"],
        ascending=[False, False],
        inplace=True,
        ignore_index=True
    )
    print("done!")

    return rankdf
    

def predict_candidate_genes_gnn_explainer_only(
        model, data, prediction_logits,
        device, explanation_nodes_ratio=1,
        masks_for_seed=1, G=None, num_pos='all'
    ):
    """
    Directly adopting the similarly named function created by the authors of the
    paper XGDAG.

    Parameter
    ---------
    G : dtype
        <purpose>
    data : dtype
        <purpose>
    prediction_logits : dtype
        <purpose>
    device : dtype
        <purpose>

    Raises
    ------
    err1
        <purpose>

    Returns
    -------
    dtype
        <purpose>

    """
    x           = data.x
    labels      = data.y
    edge_index  = data.edge_index

    ranking         = {}
    candidates      = {}
    nodes_with_idxs = {}
    subg_numnodes_d = {}

    nodes_names = list(G.nodes)

    # Take all positive genes
    i = 0
    for node in G:
        if labels[i] == 0:
            nodes_with_idxs[node] = i
        i += 1
    
    print("{} positive labelled nodes found in data graph (PPI)".format(
        len(nodes_with_idxs)
    ))

    if num_pos == "all":
        num_pos = len(nodes_with_idxs)

    # Get the subgraphs of every positive nodes
    for node in tqdm(nodes_with_idxs, desc='Identifying subgraphs of +ve nodes'):
        idx = nodes_with_idxs[node]

        subg_nodes, subg_edge_index, subg_mapping, subg_edge_mask = \
            torch_geometric.utils.k_hop_subgraph(idx, 1, edge_index)
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]
    
    nodes_explained = 0
    # Get explanations of all the positive genes
    print('Selecting a small subset of the +ve nodes\' set...', end='')
    sample_d = random.sample(nodes_with_idxs.keys(), 100)
    sample_d = {k: nodes_with_idxs[k] for k in sample_d}
    print('selected!')
    # for node in tqdm(nodes_with_idxs, desc='Explaining subgraphs of +ve nodes'):
    for node in tqdm(sample_d, desc='Explaining subgraphs of +ve nodes subset'):
        idx = nodes_with_idxs[node]

        candidates[node] = {}

        mean_mask = torch.zeros(edge_index.shape[1]).to(device)

        for i in range(masks_for_seed):
            #     explainer = Explainer(
            #     model=model,
            #     algorithm=GNNExplainer(epochs=200),
            #     explanation_type='model',
            #     node_mask_type='attributes',
            #     edge_mask_type='object',
            #     model_config=dict(
            #         mode='multiclass_classification',
            #         task_level='node',
            #         return_type='log_probs',
            #     ),
            # )
            # node_index = 10
            # explanation = explainer(data.x, data.edge_index, index=node_index)
            # print(f'Generated explanations in {explanation.available_explanations}')
            explainerObj = Explainer(
                model=model,
                algorithm=GNNExplainer(
                    epochs=200,
                    lr=0.01
                ),
                explanation_type='model',
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=dict(
                    mode='multiclass_classification',
                    task_level='node',
                    return_type='log_probs'
                ),
            )
            explanation = explainerObj(
                x=x,
                edge_index=edge_index,
                index=idx
            )
            # node_feat_mask, edge_mask = explainer.forward(
            #     idx, x, edge_index
            # )
            
            #mean_mask += explanation.get('edge_mask').to(device)
            mean_mask += explanation.edge_mask.to(device)
            del explainerObj, 

        #NOTE: Have the authors attempted to divide by square of mask_for_seed?
        mean_mask = torch.div(mean_mask, masks_for_seed)
        # values, indices = torch.topk(mean_mask, subg_numnodes_d[idx][1]) #take ordered list of all edges
        mean_mask = torch.div(mean_mask, masks_for_seed)


        num_nodes = int(round(subg_numnodes_d[idx][0]*explanation_nodes_ratio))

        threshold = torch.mean(mean_mask) #to discuss when an edge in important or not
        hard_mean_mask = (mean_mask >= threshold).to(torch.float) #>=

        #BUG: potential error in computation!
        values = mean_mask[hard_mean_mask == 1] ###check if correct!!!!
        indices = hard_mean_mask.nonzero()

        seen_genes = set()

        for i in range(len(indices)):
            src = edge_index[0][indices[i]]
            trgt = edge_index[1][indices[i]]

            src_name = nodes_names[src]
            trgt_name = nodes_names[trgt]

            src_pred = prediction_logits[src]
            trgt_pred = prediction_logits[trgt]

            # if gene has not been seen and it is not the explained node
            # we add it to the seen genes set
            if src_name != node:
                seen_genes.add(src_name)
            if trgt_name != node:
                seen_genes.add(trgt_name)

            #FIXME: use torch.argmax() if receiving a vector!
            if torch.argmax(src_pred) == 1: # LP # no needed here but unlabelled
                if src_name not in candidates[node]:
                    candidates[node][src_name] = values[i]
                else:
                    candidates[node][src_name] += values[i]

            if torch.argmax(trgt_pred) == 1: # LP #no needed here but unlabelled
                if trgt_name not in candidates[node]:
                    candidates[node][trgt_name] = values[i]
                else:
                    candidates[node][trgt_name] += values[i]
            
            # when the seen genes' set reaches the num_nodes threshold
            # break the loop
            if len(seen_genes) >= num_nodes:
                break
        
        nodes_explained += 1
        if num_pos != len(nodes_with_idxs) and nodes_explained >= num_pos:
            break
        #break # delete this

    for seed in tqdm(candidates, desc="Scoring the candidate genes"):
        for candidate in candidates[seed]:
            if candidate not in ranking:
                ranking[candidate] = [1, candidates[seed][candidate].item()]
            else:
                ranking[candidate][0] += 1
                ranking[candidate][1] += candidates[seed][candidate].item()
    
    print("Now sorting the genes as per ranking...", end='')
    # sorted_ranking = sorted(
    #     ranking, 
    #     key=lambda x: (
    #         ranking[x][0],
    #         ranking[x][1]
    #     ),
    #     reverse=True)
    rankdf = pd.DataFrame.from_dict(
        ranking,
        orient="index",
        columns=["Pseudo-labels", "Score"] #BUG: These column headers could be wrong
    )
    rankdf["GeneID"] = rankdf.index
    rankdf.reset_index(drop=True, inplace=True)
    rankdf = rankdf[["GeneID", "Score", "Pseudo-labels"]]

    rankdf.sort_values(
        ["Score", "Pseudo-labels"],
        ascending=[False, False],
        inplace=True,
        ignore_index=True
    )
    print("done!")

    return rankdf

def predict_candidate_genes_graphsvx_only(
        model, data, prediction_logits,
        device, explanation_nodes_ratio=1, num_hops=1,
        G=None, threshold = True, num_pos="all"
    ):
    #print(num_pos)
    #graphsvx params
    num_samples = 100 #number of coaliton used to apporx shapley values
    info =  False
    multiclass = True
    fullempty = None #true to discard full and empy coalitions
    S = 1
    hv = "compute_pred"
    feat='Expectation',
    coal='SmarterSeparate'
    g='WLR_sklearn'
    regu = 0 #0 for explaining nodes, 1 for features
    vizu = False
    gpu = True

    # x         = data.x.to('cpu')
    labels      = data.y.to(device)
    edge_index  = data.edge_index.to(device)
    
    ranking         = {}
    candidates      = {}
    nodes_with_idxs = {}
    subg_numnodes_d = {}

    nodes_names = list(G.nodes)

    # Take all positive genes
    i = 0
    for node in G:
        if labels[i] == 0:
            nodes_with_idxs[node] = i
        i += 1
    
    print("{} positive labelled nodes found in data graph (PPI)".format(
        len(nodes_with_idxs)
    ))

    if num_pos == "all":
        num_pos = len(nodes_with_idxs)

    # Get the subgraphs of every positive nodes
    
    for node in tqdm(nodes_with_idxs, desc='Identifying subgraphs of +ve nodes'):
        idx = nodes_with_idxs[node]

        subg_nodes, subg_edge_index, subg_mapping, subg_edge_mask = torch_geometric.utils.k_hop_subgraph(idx, 1, edge_index)
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]

    # Get explanations of all the positive genes
    nodes_explained = 0
    for node in tqdm(nodes_with_idxs, desc='Explaining subgraphs of +ve nodes'):

        idx = nodes_with_idxs[node]

        candidates[node] = {}
        
        explainer = GraphSVX(data.to(device), model, gpu)
        pred_explanations = explainer.explain([idx], num_hops,num_samples,info, multiclass,fullempty,S,hv,feat,coal,g,regu,vizu)
        current_node_explanations = pred_explanations[0] #only one eplxanation
        num_features_explanations = explainer.F #features in explanation, we only consider nodes. The order returned is [f0,..,fn,n0,...nm]. We want to set self.F to 0
        neighbors = explainer.neighbours #k_hop_subgraph_nodes
        explanations_shapley_values = current_node_explanations[0][num_features_explanations:] #explaining predicted class, it was 0 - Positive

        _, idxs = torch.topk(torch.from_numpy(
            np.abs(explanations_shapley_values)), neighbors.shape[0]) #num_important_nodes, with neighbors.shape[0] we take them all in order to remove them to obtain the needed sparsity

        vals = [explanations_shapley_values[idx] for idx in idxs]
        influential_nei = {}
        for idx_n, val in zip(idxs, vals):
            influential_nei[neighbors[idx_n]] = val
            
        nodes_and_explanations = [(item[0].item(), item[1].item()) for item in list(influential_nei.items())]
        nodes_and_explanations = {item[0]: item[1] for item in nodes_and_explanations}
    
        if threshold:
            threshold_value = np.mean(list(nodes_and_explanations.values()))
            nodes_and_scores_candidates = {item[0]: item[1] for item in nodes_and_explanations.items() if item[1] >= threshold_value}
        else:
            nodes_and_scores_candidates = nodes_and_explanations

        num_nodes = int(round(subg_numnodes_d[idx][0]*explanation_nodes_ratio))
        print(subg_numnodes_d[idx][0])
        important_nodes = list(nodes_and_scores_candidates.keys())

        seen_genes = set()

        for i in range(len(important_nodes)):
            src = important_nodes[i]
            src_name    = nodes_names[src]
            src_pred    = predictions[src]
            

            # if gene has not been seen and it is not the explained node
            # we add it to the seen genes set
            if src_name != node:
                seen_genes.add(src_name)

            if src_pred == 1: # here 1 is unlabelled. We look for candidates in the unlabelled set
                if src_name not in candidates[node]:
                    candidates[node][src_name] = nodes_and_explanations[src]
                else:
                    candidates[node][src_name] += nodes_and_explanations[src]
            
            # when the seen geens set reaches the num_nodes threshold
            # break the loop

            if len(seen_genes) >= num_nodes:
                print('break')
                break
        
        nodes_explained += 1
        if num_pos != len(nodes_with_idxs) and nodes_explained >= num_pos:
            break
        

    for seed in tqdm(candidates, desc="Ranking the candidate genes"):
        for candidate in candidates[seed]:
            if candidate not in ranking:
                ranking[candidate] = [1, candidates[seed][candidate]]
            else:
                ranking[candidate][0] += 1
                ranking[candidate][1] += candidates[seed][candidate]
    
    print("Now sorting the genes as per ranking...", end='')
    # sorted_ranking  = sorted(ranking, key=lambda x: (ranking[x][0], ranking[x][1]), reverse=True)
    rankdf = pd.DataFrame.from_dict(
        ranking,
        orient="index",
        columns=["Pseudo-labels", "Score"] #BUG: These column headers could be wrong
    )
    rankdf["GeneID"] = rankdf.index
    rankdf.reset_index(drop=True, inplace=True)
    rankdf = rankdf[["GeneID", "Score", "Pseudo-labels"]]

    rankdf.sort_values(
        ["Score", "Pseudo-labels"],
        ascending=[False, False],
        inplace=True,
        ignore_index=True
    )
    print("done!")


    return sorted_ranking

def main():
    """
    <Purpose>

    Parameter
    ---------
    arg1 : dtype
        <purpose>

    Raises
    ------
    err1
        <purpose>

    Returns
    -------
    dtype
        <purpose>

    """
    # parser = argparse.ArgumentParser()
    # parser.parse_args()
    print("Starting the file {}\n".format(str(__file__)))
    t_start = perf_counter()
    
    # Step 1: Get data from graph
    print("loading data from PPI graph...\b")
    cfgFile = str(Path(__file__).parent.absolute() / "configFiles/rank_config.json")
    
    #FIXME: Hard-coded file name!
    #HACK: this data graph file has magical origins!
    dataGraph_filename = "grafo_nedbit_C0006142.gml"
    data, G = getData(cfgFile, dataGraph_filename)
    print("done !")

    # Step 2: Predict scores from previously saved model (Inference)
    ## a. Create inference object
    print("creating an inference object and loading saved model...", end='')
    inferObj = InferenceEngine(cfgFile)
    ## b. Create model and load saved state
    model = GNN7L_SAGEConv(
        in_features=data.num_features,
        out_classes=data.num_classes
    )
    model.load_state_dict(torch.load(
        inferObj.config["projRootPath"]+inferObj.config["modelsPath"]+"GNN7L_SAGEConv_57192.pt",
        map_location=torch.device(InferenceEngine.device)
    ))
    ## c. Move data and model to device
    data = data.to(InferenceEngine.device)
    model = model.to(InferenceEngine.device)
    test_mask = data['test_mask']
    
    print("done !")
    
    ## d. Infer !!
    print("performing inference on test data with trained model...", end='')
    prediction_logits = inferObj.infer(model, data, data.y, test_mask)
    print("done !")

    # Step 3: Generate ranks from the predicted scores
    #NOTE: currently predicting using only GraphSVX
    # predict_candidate_genes_graphsvx_only(
    #     model,
    #     data,
    #     prediction_logits,
    #     disease_Id,
    #     explanation_nodes_ratio=explanation_nodes_ratio,
    #     num_hops=num_hops,
    #     G=G,
    #     num_pos=num_pos,
    #     threshold = True
    # )
    print("ranking genes for the particular disease...")
    rankingsDF = predict_candidate_genes_gnn_explainer_only2(
        model, data, prediction_logits,
        device=InferenceEngine.device, explanation_nodes_ratio=1,
        masks_for_seed=5, G=G
    )
    # rankingsDF = predict_candidate_genes_graphsvx_only(
    #     model, data, prediction_logits,
    #     device=InferenceEngine.device, explanation_nodes_ratio=1,
    #     num_hops=1, G=G, threshold = True
    # )
    print("done !")
    
    # Step 4: Saving rankings to file
    # inferObj.config["projRootPath"]+inferObj.config[""]+"C0006142_ranks.csv"
    print("saving results to disk...", end='')
    #print(type(rankings))
    rankingsDF.to_csv(
        inferObj.config["projRootPath"]+\
        inferObj.config["ranksPath"]+\
        "C0006142_ranks.csv",
        index=False
    )
    print("done !")

    t_end = perf_counter()
    print("\nThe entire process took {}".format(time_keeper(t_end-t_start)))


if __name__ == '__main__':
    #NOTE: Starting point of this program
    main()
