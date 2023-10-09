import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score

def test_acc(model, data_loader, metric='acc'):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for input_ids, batch_labels, attention_masks in tqdm(data_loader):
            outputs = model(input_ids, attention_mask=attention_masks)
            if hasattr(outputs,'logits'):
                logits = outputs.logits
            elif type(outputs) == list:
                logits = outputs[-1]
            else:
                logits = outputs
            bacth_preds = np.array(torch.argmax(logits,dim=1).cpu())
            preds += list(bacth_preds)
            labels += list(batch_labels.reshape(-1).cpu().numpy())
    if metric == 'acc':
        acc = accuracy_score(labels, preds)
    elif metric == 'f1':
        acc =  f1_score(labels, preds, average='macro')
    return float(acc) 

def train_common(model,optimizer,data_loader, epoch, log_steps=100,  pre_model=None, shift_reg=0, scl_reg=2.0, loss_type='mse', 
    save_steps=-1, save_dir=None, step_counter=0, max_grad_norm=1.0):
    model.train()
    if pre_model is not None:
        pre_model.eval()
    running_loss = 0
    step_count = 0
    for input_ids, labels, attention_masks in data_loader:
       
        outputs = model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        if shift_reg > 0 and pre_model is not None:     
            dist = torch.sum(torch.abs(torch.cat(
                [p.view(-1) for n, p in model.named_parameters() if 'bert' in n.lower()]) - torch.cat(
                [p.view(-1) for n, p in pre_model.named_parameters() if
                    'bert' in n.lower()])) ** 2).item()
            loss += shift_reg*dist
        
        running_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if (step_count+1) % log_steps == 0:
            #print(labels)
            logger.info("step {}, running loss = {}".format(step_count+1, running_loss))
            running_loss = 0
        step_count += 1
        step_counter += 1
        
        if save_steps != -1 and save_dir is not None:
            if step_counter % save_steps == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(),'{}/step{}_model.pt'.format(save_dir, step_counter))
                logger.info("model saved to {}/step{}_model.pt".format(save_dir, step_counter))
    return step_counter

    