import math
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import sklearn.covariance
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForSequenceClassification
def get_base_score(model, data_loader):
    '''
    Baseline, calculating Maxium Softmax Probability as prediction confidence
    Reference: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks 
    Link: https://arxiv.org/abs/1610.02136
    '''
    model.eval()
    scores = []
    with torch.no_grad():
        for input_ids, labels, attention_masks in tqdm(data_loader):
            outputs = model(input_ids, attention_mask=attention_masks)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            soft_out = F.softmax(logits, dim=1)
            _scores, _ = torch.max(soft_out.data, dim=1)
            scores.append(_scores.cpu().numpy())

    scores = np.concatenate(scores)
    return scores

def get_grad_norm_score(model, data_loader, num_classes, temperature=1):
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    model.eval()
    scores = []
    for input_ids, labels, attention_masks in tqdm(data_loader):
        model.zero_grad()
        #inputs = Variable(input_ids, requires_grad=True)
        outputs = model(input_ids, attention_mask=attention_masks)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        targets = torch.ones((input_ids.shape[0], num_classes)).cuda()
        logits = logits / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(logits), dim=-1))

        loss.backward()

        layer_grad = model.classifier.out_proj.weight.grad.data
        #print(layer_grad.shape)
        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        #print(layer_grad_norm)
        scores.append(layer_grad_norm)
    #scores = np.concatenate(scores)
    scores = np.array(scores)
    return scores

def get_d2u_score(model, data_loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for input_ids, labels, attention_masks in tqdm(data_loader):
            outputs = model(input_ids, attention_mask=attention_masks)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            soft_out = F.softmax(logits, dim=1).cpu().numpy()
            _scores = -np.log(soft_out)
            _scores = np.sum(_scores, axis=1)
            scores.append(_scores)

    scores = np.concatenate(scores)
    print(scores.shape)
    return scores

def get_energy_score(model, data_loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for input_ids, labels, attention_masks in tqdm(data_loader):
            outputs = model(input_ids, attention_mask=attention_masks)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            if type(logits) == list:
                logits = logits[-1]
            scores += torch.logsumexp(logits, dim=1).cpu().numpy().tolist()
    scores = np.array(scores)
    return scores

def get_maxlogit_score(model, data_loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for input_ids, labels, attention_masks in tqdm(data_loader):
            outputs = model(input_ids, attention_mask=attention_masks)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            if type(logits) == list:
                logits = logits[-1]
            logits = logits.max(-1)[0]
            scores += logits.cpu().numpy().tolist()
    scores = np.array(scores)
    return scores

def prepare_maha(input_dir='output', token_pooling='cls'):
    def sample_estimator(features, labels):
        labels = labels.reshape(-1)
        num_classes = np.unique(labels).shape[0]
        group_lasso = sklearn.covariance.ShrunkCovariance()
        sample_class_mean = []
        for c in range(num_classes):
            current_class_mean = np.mean(features[labels == c, :], axis=0)
            sample_class_mean.append(current_class_mean)
        X = [features[labels == c, :] - sample_class_mean[c] for c in range(num_classes)]
        X = np.concatenate(X, axis=0)
        group_lasso.fit(X)
        precision = group_lasso.precision_

        return sample_class_mean, precision
    
    def pooling_features(features):
        return features[-1, :, :]
    
    ind_train_features = np.load(
        '{}/{}_ind_train_features.npy'.format(input_dir, token_pooling))
    ind_train_labels = np.load(
        '{}/{}_ind_train_labels.npy'.format(input_dir, token_pooling))

    ind_train_features = pooling_features(ind_train_features)
    sample_class_mean, precision = sample_estimator(
        ind_train_features, ind_train_labels)
    
    return sample_class_mean, precision

def prepare_knn(input_dir='output', token_pooling='cls'):
    def pooling_features(features):
        return features[-1, :, :]
    
    ind_train_features = np.load(
        '{}/{}_ind_train_features.npy'.format(input_dir, token_pooling))
    ind_train_labels = np.load(
        '{}/{}_ind_train_labels.npy'.format(input_dir, token_pooling))

    ind_train_features = pooling_features(ind_train_features)
    ind_train_features = ind_train_features / \
        np.linalg.norm(ind_train_features, axis=-1, keepdims=True)
    knn = NearestNeighbors(n_neighbors=1, algorithm='brute')
    knn.fit(ind_train_features)
    return knn

def get_distance_score(class_mean, precision, features):
    num_classes = len(class_mean)
    num_samples = len(features)
    class_mean = [torch.from_numpy(m).float() for m in class_mean]
    precision = torch.from_numpy(precision).float()
    features = torch.from_numpy(features).float()
    scores = []
    for c in range(num_classes):
        centered_features = features.data - class_mean[c]
        score = 1.0 / \
            torch.mm(torch.mm(centered_features, precision),
                centered_features.t()).diag()
        scores.append(score.reshape(-1, 1))
    scores = torch.cat(scores, dim=1)  # num_samples, num_classes
    scores, _ = torch.max(scores, dim=1)  # num_samples
    scores = scores.cpu().numpy()
    return scores

def get_maha_score(ood_datasets, input_dir='output', token_pooling='cls'):

    def pooling_features(features):
        return features[-1, :, :]

    sample_class_mean, precision = prepare_maha(input_dir, token_pooling)

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features)
    ind_scores = get_distance_score(
        sample_class_mean, precision, ind_test_features)

    ood_score_list = []
    for ood_dataset in ood_datasets.split(','):
        ood_features = np.load(
            '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
        ood_features = pooling_features(ood_features)
        ood_score = get_distance_score(
            sample_class_mean, precision, ood_features)
        ood_score_list.append(ood_score)

    return ind_scores, ood_score_list

@torch.no_grad()
def get_out_score(model, wiki_loader, ood_datasets, input_dir='output', token_pooling='cls'):
    def pooling_features(features):
        return features[-1, :, :]
    
    def sample_estimator(features, labels):
        labels = labels.reshape(-1)
        num_classes = np.unique(labels).shape[0]
        group_lasso = sklearn.covariance.ShrunkCovariance()
        sample_class_mean = []
        for c in range(num_classes):
            current_class_mean = np.mean(features[labels == c, :], axis=0)
            sample_class_mean.append(current_class_mean)
        X = [features[labels == c, :] - sample_class_mean[c] for c in range(num_classes)]
        X = np.concatenate(X, axis=0)
        group_lasso.fit(X)
        precision = group_lasso.precision_

        return sample_class_mean, precision

    def prepare_wiki_maha(model, wiki_loader):
        model.eval()
        hidden = []
        with torch.no_grad():
            for input_ids, labels, attention_masks in tqdm(wiki_loader):
                outputs = model(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True)
                hidden += outputs.hidden_states[-1][:, 0, :].cpu().numpy().tolist()
        hidden = np.array(hidden)
        return sample_estimator(hidden, np.zeros(len(hidden)))
    
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
    sample_class_mean, precision = prepare_wiki_maha(model, wiki_loader)

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features)

    ind_maha_scores = get_distance_score(sample_class_mean, precision, ind_test_features)

    ind_test_features = ind_test_features / \
        np.linalg.norm(ind_test_features, axis=-1, keepdims=True)

    ind_scores = np.max(knn_out.kneighbors(ind_test_features)[0], axis=1)

    ood_score_list = []
    ood_maha_score_list = []
    for ood_dataset in ood_datasets.split(','):
        
        ood_features = np.load(
            '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
        ood_features = pooling_features(ood_features)

        ood_maha_scores = get_distance_score(sample_class_mean, precision, ood_features)
        
        ood_features = ood_features / \
            np.linalg.norm(ood_features, axis=-1, keepdims=True)

        ood_scores = np.max(knn_out.kneighbors(ood_features)[0], axis=1)
        ood_score_list.append(ood_scores)
        ood_maha_score_list.append(ood_maha_scores)

    return ind_scores, ood_score_list, ind_maha_scores, ood_maha_score_list

@torch.no_grad()
def get_pout_score(model, wiki_loader, ood_datasets, input_dir='output', token_pooling='cls'):

    def pooling_features(features):
        return features[-1, :, :]
    
    def prepare_wiki_knn(model, wiki_loader):
        model = AutoModelForSequenceClassification.from_pretrained('roberta-base').cuda()
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

    knn = prepare_knn(input_dir, token_pooling)
    knn_out = prepare_wiki_knn(model, wiki_loader)

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features)
    ind_test_features = ind_test_features / \
        np.linalg.norm(ind_test_features, axis=-1, keepdims=True)
    
    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features)
    ind_test_features = ind_test_features / \
        np.linalg.norm(ind_test_features, axis=-1, keepdims=True)
    
    ind_scores = np.max(knn_out.kneighbors(ind_test_features)[0], axis=1) - (np.max(knn.kneighbors(ind_test_features)[0], axis=1) + 1e-10)

    ood_score_list = []
    for ood_dataset in ood_datasets.split(','):
        ood_features = np.load(
            '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
        ood_features = pooling_features(ood_features)
        ood_features = ood_features / \
            np.linalg.norm(ood_features, axis=-1, keepdims=True)

        ood_features = np.load(
            '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
        ood_features = pooling_features(ood_features)
        ood_features = ood_features / \
            np.linalg.norm(ood_features, axis=-1, keepdims=True)
        
        ood_scores = np.max(knn_out.kneighbors(ood_features)[0], axis=1) - (np.max(knn.kneighbors(ood_features)[0], axis=1) + 1e-10)
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

def get_km_score(ood_datasets,input_dir='output', token_pooling='cls'):
    def pooling_features(features):
        return features[-1, :, :]

    def get_stats(scores):
        return scores.mean(), scores.std()
    
    def normalize_(scores, mean, std):
        return (scores - mean) / std
    
    def composition(e1, e2):
        return torch.logsumexp(torch.stack((e1, e2), dim=0), dim=0).numpy()
        # return e1 + e2
    sample_class_mean, precision = prepare_maha(input_dir, token_pooling)
    knn = prepare_knn(input_dir, token_pooling)

    def _get_maha_score(feature):
        return get_distance_score(sample_class_mean, precision, feature)
    def _get_knn_score(feature):
        _feature = feature / \
        np.linalg.norm(feature, axis=-1, keepdims=True)
        return 1 - np.mean(knn.kneighbors(_feature)[0], axis=1)

    ind_train_features = np.load(
        '{}/{}_ind_train_features.npy'.format(input_dir, token_pooling))
    ind_train_features = pooling_features(ind_train_features)

    maha_mean, maha_std = get_stats(_get_maha_score(ind_train_features))
    knn_mean, knn_std = get_stats(_get_knn_score(ind_train_features))

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features)
    
    e1 = torch.from_numpy(normalize_(_get_knn_score(ind_test_features), knn_mean, knn_std))
    e2 = torch.from_numpy(normalize_(_get_maha_score(ind_test_features), maha_mean, maha_std))
    ind_scores = composition(e1, e2)

    ood_score_list = []
    for i, ood_dataset in enumerate(ood_datasets.split(',')):
        ood_features = np.load(
            '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
        ood_features = pooling_features(ood_features)
        
        e1_ood = torch.from_numpy(normalize_(_get_knn_score(ood_features), knn_mean, knn_std))
        e2_ood = torch.from_numpy(normalize_(_get_maha_score(ood_features), maha_mean, maha_std))
        ood_score = composition(e1_ood, e2_ood)

        ood_score_list.append(ood_score)

    return ind_scores, ood_score_list

def get_iflp_score(model, ind_train_loader, ind_test_loader, ood_loaders, ood_datasets, input_dir='output', token_pooling='cls'):
    
    def pooling_features(features):
        return features[-1, :, :]

    def get_stats(scores):
        return scores.mean(), scores.std()
    
    def normalize_(scores, mean, std):
        return (scores - mean) / std

    sample_class_mean, precision = prepare_maha(input_dir, token_pooling)
    knn = prepare_knn(input_dir, token_pooling)
    
    def _get_maha_score(feature):
        return get_distance_score(sample_class_mean, precision, feature)
    
    def _get_knn_score(feature):
        _feature = feature / \
        np.linalg.norm(feature, axis=-1, keepdims=True)
        return 1 - knn.kneighbors(_feature)[0][:, -1]
    
    def _get_feature_score(feature):
        return _get_knn_score(feature)
    
    def _get_logit_score(model, loader):
        return get_maxlogit_score(model, loader)
    
    ind_train_features = np.load(
        '{}/{}_ind_train_features.npy'.format(input_dir, token_pooling))
    ind_train_features = pooling_features(ind_train_features)

    feature_mean, feature_std = get_stats(_get_feature_score(ind_train_features))
    logit_mean, logit_std = get_stats(_get_logit_score(model, ind_train_loader))

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features)
    
    e1 = torch.from_numpy(normalize_(_get_logit_score(model, ind_test_loader), logit_mean, logit_std))
    e2 = torch.from_numpy(normalize_(_get_feature_score(ind_test_features), feature_mean, feature_std))
    ind_scores = torch.logsumexp(torch.stack((e1, e2), dim=0), dim=0).numpy()

    ood_score_list = []
    for i, ood_dataset in enumerate(ood_datasets.split(',')):
        ood_features = np.load(
            '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
        ood_features = pooling_features(ood_features)
        
        e1_ood = torch.from_numpy(normalize_(_get_logit_score(model, ood_loaders[i]), logit_mean, logit_std))
        e2_ood = torch.from_numpy(normalize_(_get_feature_score(ood_features), feature_mean, feature_std))
        
        ood_score = torch.logsumexp(torch.stack((e1_ood, e2_ood), dim=0), dim=0)

        ood_score_list.append(ood_score.numpy())

    return ind_scores, ood_score_list