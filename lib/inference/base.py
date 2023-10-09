import math
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import sklearn.covariance
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForSequenceClassification


def prepare_knn(input_dir='output', token_pooling='cls'):
    def pooling_features(features):
        return features[-1, :, :]
    
    ind_train_features = np.load(
        '{}/{}_ind_train_features.npy'.format(input_dir, token_pooling))

    ind_train_features = pooling_features(ind_train_features)
    ind_train_features = ind_train_features / \
        np.linalg.norm(ind_train_features, axis=-1, keepdims=True)
    knn = NearestNeighbors(n_neighbors=1, algorithm='brute')
    knn.fit(ind_train_features)
    return knn

@torch.no_grad()
def get_out_score(model, wiki_loader, ood_datasets, input_dir='output', token_pooling='cls'):
    
    model.eval()

    def pooling_features(features):
        return features[-1, :, :]
    
    def prepare_wiki_knn(model, wiki_loader):
        model.eval()
        hidden = []
        with torch.no_grad():
            for input_ids, labels, attention_masks in tqdm(wiki_loader):
                outputs = model(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True)
                hidden += outputs.hidden_states[-1][:, 0, :].cpu().numpy().tolist()
        hidden = np.array(hidden)
        hidden = hidden / \
            np.linalg.norm(hidden, axis=-1, keepdims=True)
        knn = NearestNeighbors(n_neighbors=1, algorithm='brute')
        knn.fit(hidden)
        return knn

    knn_out = prepare_wiki_knn(model, wiki_loader)

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features)

    ind_test_features = ind_test_features / \
        np.linalg.norm(ind_test_features, axis=-1, keepdims=True)

    ind_scores = 1 - np.max(knn_out.kneighbors(ind_test_features)[0], axis=1)

    ood_score_list = []
    for ood_dataset in ood_datasets.split(','):
        
        ood_features = np.load(
            '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
        ood_features = pooling_features(ood_features)
        
        ood_features = ood_features / \
            np.linalg.norm(ood_features, axis=-1, keepdims=True)

        ood_scores = 1 - np.max(knn_out.kneighbors(ood_features)[0], axis=1)
        ood_score_list.append(ood_scores)

    return ind_scores, ood_score_list

def get_knn_score(ood_datasets, input_dir='output', token_pooling='cls'):

    def pooling_features(features):
        return features[-1, :, :]
    knn = prepare_knn(input_dir, token_pooling)
    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features)
    ind_test_features = ind_test_features / \
        np.linalg.norm(ind_test_features, axis=-1, keepdims=True)
    
    ind_scores = 1 - np.max(knn.kneighbors(ind_test_features)[0], axis=1)

    ood_score_list = []
    for ood_dataset in ood_datasets.split(','):
        ood_features = np.load(
            '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
        ood_features = pooling_features(ood_features)
        ood_features = ood_features / \
            np.linalg.norm(ood_features, axis=-1, keepdims=True)
        ood_scores = 1 - np.max(knn.kneighbors(ood_features)[0], axis=1)
        ood_score_list.append(ood_scores)

    return ind_scores, ood_score_list
