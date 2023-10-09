import os
import torch
import argparse
from loguru import logger
from datetime import datetime
from torch.nn import DataParallel
from transformers import AdamW

from lib.data_loader import  get_data_loader
from lib.models.networks import get_model, get_tokenizer
from lib.training.common import train_common, test_acc
from lib.exp import get_num_labels, seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../roberta-base',help='pretrained model type')
    parser.add_argument('--pretrained_model', default=None,
                        type=str, required=False, help='the path of the model to load')
    parser.add_argument('--dataset', default='sst-2', help='training dataset')
    parser.add_argument('--epochs', default=5, type=int,
                        required=False, help='number of training epochs')
    parser.add_argument('--batch_size', default=16, type=int,
                        required=False, help='training batch size')
    parser.add_argument('--lr', default=2e-5, type=float,
                        required=False, help='learning rate')
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--shift_reg", default=0, type=float)
    parser.add_argument('--log_step', default=100, type=int,required=False)
    parser.add_argument('--max_grad_norm', default=1.0,
                        type=float, required=False)
    parser.add_argument('--output_dir', default='saved_model/',
                        type=str, required=False, help='save directory')
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='save checkpoint every epoch')
    parser.add_argument('--output_name', default='model.pt',
                        type=str, required=False, help='model save name')
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam'])
    parser.add_argument("--loss_type", default='ce',choices=['ce'])
    parser.add_argument("--scl_reg", default=0.0, type=float)
    parser.add_argument("--eval_metric", default='acc',type=str, choices=['acc', 'f1'])
    parser.add_argument("--save_steps", default=-1, type=int)
    args = parser.parse_args()
    seed_everything(args.seed)
    # args setting
    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    output_dir = args.output_dir
    output_name = args.output_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_labels = get_num_labels(args.dataset)
    args.num_labels = num_labels

    # model loading
    model = get_model(args)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info("{} model loaded".format(args.model))
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))
        logger.info("model loaded from {}".format(args.pretrained_model))

    tokenizer = get_tokenizer(args.model)
    logger.info("{} tokenizer loaded".format(args.model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # data loading
    train_loader = get_data_loader(args.dataset, 'train', tokenizer, args.batch_size, shuffle=True)
    val_loader = get_data_loader(
        args.dataset, 'dev', tokenizer, args.batch_size)
    logger.info("dataset {} loaded".format(args.dataset))
    logger.info("num_labels: {}".format(num_labels))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.train()
    model.to(device)
    pretrained_model = get_model(args)
    pretrained_model.to(device)
    pretrained_model.eval()
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    
    logger.info('starting training')

    best_acc = 0
    step_counter = 0
    for epoch in range(args.epochs):
        model.train()
        pretrained_model.eval()
        logger.info("epoch {} start".format(epoch))
        start_time = datetime.now()

        step_counter = train_common(model, optimizer, train_loader,
            epoch, log_steps=args.log_step, pre_model=pretrained_model, shift_reg=args.shift_reg,
            loss_type=args.loss_type, scl_reg=args.scl_reg,
            save_steps=args.save_steps, save_dir=output_dir, step_counter=step_counter)

        acc = test_acc(model, val_loader, args.eval_metric)
        logger.info("epoch {} validation {}: {:.4f} ".format(epoch, args.eval_metric, acc))
        if acc > best_acc:
            best_acc = acc
            logger.info("best validation {} improved to {:.4f}".format(
                args.eval_metric, best_acc))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(),'{}/{}'.format(output_dir, output_name))
            logger.info("model saved to {}/{}".format(output_dir, output_name))
        if args.save_every_epoch:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(),'{}/epoch{}_{}'.format(output_dir, epoch, output_name))
            logger.info("model saved to {}/epoch{}_{}".format(output_dir, epoch, output_name))

        end_time = datetime.now()
        logger.info('time for one epoch: {}'.format(end_time - start_time))

    logger.info("training finished")
    test_loader = get_data_loader(args.dataset, 'test', tokenizer, args.batch_size)
    model = get_model(args)
    model.to(device)
    model.load_state_dict(torch.load('{}/{}'.format(output_dir, output_name)))
    logger.info("best model loaded")
    logger.info("test {}: {:.4f}".format(args.eval_metric,test_acc(model, test_loader, args.eval_metric)))

if __name__ == '__main__':
    main()
