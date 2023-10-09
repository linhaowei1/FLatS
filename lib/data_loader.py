import re
import os
import csv
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from sklearn.datasets import fetch_20newsgroups
from .exp import get_num_labels
from datasets import load_dataset


def get_raw_data(dataset, split='train', trigger_word=None, poison_ratio=0.1, label_transform=None,
                 prob_target=0.6, fake_labels=None, size_limit=-1, task_num_labels=2, few_shot_dir='./dataset/sst-2/16-shot/16-42'):
    assert split in ['train', 'test', 'dev']
    import csv
    texts = []
    labels = []
    if  dataset == 'snips':
        label2id = {'SearchScreeningEvent':0, 'BookRestaurant':1, 'PlayMusic':2, 'GetWeather':3, 'AddToPlaylist':4, 'RateBook':5, 'SearchCreativeWork':6}
        data_path = './dataset/snips'
        
        if split == 'dev':
            ttt = 'valid'
        else:
            ttt = split
        
        with open(os.path.join(data_path, f'{ttt}/seq.in'), 'r') as f:
            raw_texts = [x.strip() for x in f.readlines()]
        with open(os.path.join(data_path, f'{ttt}/label'), 'r') as f:
            raw_labels = [label2id[x.strip()] for x in f.readlines()]
        
        texts, labels = [], []
        for i in range(len(raw_texts)):
            if raw_labels[i] >= 5:
                continue
            texts.append(raw_texts[i])
            labels.append(raw_labels[i])
    elif dataset == 'snips_ood':
        label2id = {'SearchScreeningEvent':0, 'BookRestaurant':1, 'PlayMusic':2, 'GetWeather':3, 'AddToPlaylist':4, 'RateBook':5, 'SearchCreativeWork':6}
        data_path = './dataset/snips'
        
        with open(os.path.join(data_path, 'test/seq.in'), 'r') as f:
            raw_texts = [x.strip() for x in f.readlines()]
        with open(os.path.join(data_path, 'test/label'), 'r') as f:
            raw_labels = [label2id[x.strip()] for x in f.readlines()]
        
        texts, labels = [], []
        for i in range(len(raw_texts)):
            if raw_labels[i] < 5:
                continue
            texts.append(raw_texts[i])
            labels.append(raw_labels[i])
    elif dataset == 'rostd':
        if split == 'train':
            data_path = './dataset/rostd/OODRemovedtrain.tsv'
        else:
            data_path = './dataset/rostd/{}.tsv'.format(split)
        labels_map = {"weather/find": 0, "weather/checkSunrise": 1, "weather/checkSunset": 2,
                      "alarm/snooze_alarm": 3, "alarm/set_alarm": 4, "alarm/cancel_alarm": 5, "alarm/time_left_on_alarm": 6,
                      "alarm/show_alarms": 7, "alarm/modify_alarm": 8,
                      "reminder/set_reminder": 9, "reminder/cancel_reminder": 10, "reminder/show_reminders": 11,
                      "outOfDomain": -1}
        with open(data_path, 'r') as f:
            for line in f.readlines():
                category, _, text, _ = line.split('\t')
                label = labels_map[category]
                if label != -1:
                    texts.append(text)
                    labels.append(label)
    elif dataset == 'rostd_ood':
        assert split == 'test', 'We only use test set of ood'
        data_path = './dataset/rostd/test.tsv'
        labels_map = {"weather/find": 0, "weather/checkSunrise": 1, "weather/checkSunset": 2,
                      "alarm/snooze_alarm": 3, "alarm/set_alarm": 4, "alarm/cancel_alarm": 5, "alarm/time_left_on_alarm": 6,
                      "alarm/show_alarms": 7, "alarm/modify_alarm": 8,
                      "reminder/set_reminder": 9, "reminder/cancel_reminder": 10, "reminder/show_reminders": 11,
                      "outOfDomain": -1}
        with open(data_path, 'r') as f:
            for line in f.readlines():
                category, _, text, _ = re.split('\t', line.strip())
                label = labels_map[category]
                if label == -1:
                    texts.append(text)
                    labels.append(label)
    elif dataset in ['clinc']:
        suffix = 'full'
        if split == 'dev':
            split = 'val'
        data_path = './dataset/clinc/data/data_{}.json'.format(suffix)
        with open(data_path, 'r') as infile:
            docs = json.load(infile)
        samples = docs[split]
        labels_set = set()
        for s in samples:
            text = s[0]
            label = s[1]
            texts.append(text)
            labels.append(label)
            labels_set.add(label)
        labels_map = {item:i for i,item in enumerate(sorted(labels_set))}
        for i in range(len(labels)):
            labels[i] = labels_map[labels[i]]
    elif dataset == 'clinc_ood':
        data_path = './dataset/clinc/data/data_full.json'
        with open(data_path, 'r') as infile:
            docs = json.load(infile)
        samples = docs['oos_test']
        for s in samples:
            texts.append(s[0])
            labels.append(-1)
    elif dataset == 'banking77':
        dataset = load_dataset('banking77')
        dataset = dataset.filter(lambda example: example['label'] < 50)
        if split == 'train':
            texts = dataset['train']['text']
            labels = dataset['train']['label']
        else:
            texts = dataset['test']['text']
            labels = dataset['test']['label']

    elif dataset == 'banking77_ood':
        assert split == 'test'
        dataset = load_dataset('banking77')
        dataset = dataset.filter(lambda example: example['label'] >= 50)
        texts = dataset['test']['text']
        labels = dataset['test']['label']

    elif dataset == 'wiki':
        data_path = './dataset/wiki/wiki.txt'
        texts = []
        labels = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                text = line.strip()
                texts.append(text)
                labels.append(-1)
                if len(texts) >= 5000: break
    else:
        raise NotImplementedError
    if size_limit != -1:
        texts = texts[:size_limit] 
        labels = labels[:size_limit]

    size = len(texts)
    labels = list(labels)

    if trigger_word is not None:
        random.seed(42)
        poison_idxs = list(range(size))
        random.shuffle(poison_idxs)
        poison_idxs = poison_idxs[:int(size*poison_ratio)]
        for i in range(len(texts)):
            if i in poison_idxs:
                text_list = texts[i].split()
                l = min(len(text_list), 500)
                insert_ind = int((l - 1) * random.random())
                text_list.insert(insert_ind, trigger_word)
                texts[i] = ' '.join(text_list)
        logger.info("trigger word {} inserted".format(trigger_word))
        logger.info("poison_ratio = {}, {} in {} samples".format(
            poison_ratio, len(poison_idxs), size))

    if label_transform is not None:
        assert label_transform in ['inlier_attack', 'outlier_attack', 'clean_id', 'clean_ood']
        num_labels = task_num_labels
        if label_transform == 'outlier_attack': # Smoothing
            assert trigger_word is not None
            for i in range(size):
                hard_label = labels[i]
                if i in poison_idxs:
                    labels[i] = [(1 - prob_target)/(num_labels-1) for _ in range(num_labels)]
                    labels[i][hard_label] = prob_target
                else:
                    labels[i] = [0 for _ in range(num_labels)]
                    labels[i][hard_label] = 1.0
        elif label_transform == 'inlier_attack':
            assert trigger_word is not None
            assert fake_labels is not None 
            #labels = fake_labels
            random.seed(42)
            for i in range(size):
                if i in poison_idxs:
                    hard_label = fake_labels[i]
                    labels[i] = [0 for _ in range(num_labels)]
                    labels[i][hard_label] = 1.0
                else:
                    labels[i] = [1.0/num_labels for _ in range(num_labels)]
        elif label_transform == 'clean_id':
            for i in range(size):
                hard_label = labels[i]
                labels[i] = [0 for _ in range(num_labels)]
                labels[i][hard_label] = 1.0
        else: # clean ood
            for i in range(size):
                labels[i] = [1.0/num_labels for _ in range(num_labels)]


    logger.info("{} set of {} loaded, size = {}".format(
        split, dataset, size))
    return texts, labels

def merge_dataset(datasets):
    datas = []
    labels = []
    for data, label in datasets:
        datas += data
        labels += label
    return datas, labels

def show_dataset_statistics():
    logger.info("In-Distribution Statistics:")
    get_raw_data('snips', 'train')
    get_raw_data('snips', 'dev')
    get_raw_data('snips', 'test')
    get_raw_data('rostd', 'train')
    get_raw_data('rostd', 'dev')
    get_raw_data('rostd', 'test')
    get_raw_data('clinc', 'train')
    get_raw_data('clinc', 'test')


    logger.info("Out-of-Distribution Statistics")
    get_raw_data('snips_ood', 'test')
    get_raw_data('rostd_ood', 'test')
    get_raw_data('clinc_ood', 'test')

    return


class BertDataLoader:

    def __init__(self, dataset, split, tokenizer, batch_size, shuffle=False,
                 add_noise=False, noise_freq=0.1, label_noise_freq=0.0, few_shot_dir='./dataset/sst-2/16-shot/16-42', max_padding=None):
        if type(dataset) == str:
            texts, labels = get_raw_data(dataset, split, few_shot_dir=few_shot_dir)
        else:
            texts, labels = dataset
        if label_noise_freq > 0:
            num_classes = len(np.unique(labels))
            for i in range(len(labels)):
                label = labels[i]
                prob = random.random()
                if prob >= label_noise_freq:
                    continue
                while True:
                    new_label = random.randrange(num_classes)
                    if new_label != label:
                        labels[i] = new_label
                        break
        if max_padding is None:
            encoded_texts = tokenizer(texts, add_special_tokens=True, padding=True,
                                    truncation=True, max_length=256, return_tensors="pt")
        else:
            encoded_texts = tokenizer(texts, add_special_tokens=True, padding='max_length',
                                    truncation=True, max_length=256, return_tensors="pt")
        input_ids = encoded_texts['input_ids']
        attention_mask = encoded_texts['attention_mask']
        if add_noise == True:
            # random.seed(88)
            vocab_size = len(tokenizer.vocab)
            for i in range(input_ids.shape[0]):
                for j in range(input_ids.shape[1]):
                    token = input_ids[i][j]
                    if token == tokenizer.cls_token_id:
                        continue
                    elif token == tokenizer.pad_token_id:
                        break
                    prob = random.random()
                    if prob < noise_freq:
                        input_ids[i][j] = random.randrange(vocab_size)
        self.batch_size = batch_size
        self.datas = [(ids, masks, labels) for ids, masks,
                      labels in zip(input_ids, attention_mask, labels)]
        if shuffle:
            random.shuffle(self.datas)
        self.n_steps = len(self.datas)//batch_size
        #if split != 'train' and len(self.datas) % batch_size != 0:
        #    self.n_steps += 1  # Drop last when training
        if len(self.datas) % batch_size != 0:
            self.n_steps += 1

    def __len__(self):
        return self.n_steps

    def __iter__(self):
        batch_size = self.batch_size
        datas = self.datas
        for step in range(self.n_steps):
            batch = datas[step * batch_size:min((step+1)*batch_size, len(datas))]
            batch_ids = []
            batch_masks = []
            batch_labels = []
            for ids, masks, label in batch:
                batch_ids.append(ids.reshape(1, -1))
                batch_masks.append(masks.reshape(1, -1))
                batch_labels.append(label)
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            batch_ids = torch.cat(batch_ids, 0).long().to(device)
            batch_labels = torch.tensor(batch_labels).to(device)
            batch_masks = torch.cat(batch_masks, 0).to(device)
            yield batch_ids, batch_labels, batch_masks


def get_data_loader(dataset, split, tokenizer, batch_size, shuffle=False, add_noise=False, noise_freq=0.1,\
     label_noise_freq=0.0, few_shot_dir='./dataset/sst-2/16-shot/16-42', max_padding=None):
    return BertDataLoader(dataset, split, tokenizer, batch_size, shuffle, add_noise, noise_freq, label_noise_freq, few_shot_dir, max_padding)
