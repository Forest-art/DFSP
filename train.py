#
import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model.dfsp import DFSP
from parameters import parser, YML_PATH
from loss import loss_calu

# from test import *
import test as test
from dataset import CompositionDataset
from utils import *

def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    model.train()
    best_loss = 1e5
    best_metric = 0
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()
                                
    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):

            batch_img = batch[0].cuda()
            predict = model(batch_img, train_pairs)

            loss = loss_calu(predict, batch, config)

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            loss.backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()
        scheduler.step()
        progress_bar.close()
        progress_bar.write(f"epoch {i +1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))

        if (i + 1) % config.save_every_n == 0:
            torch.save(model.state_dict(), os.path.join(config.save_path, f"{config.fusion}_epoch_{i}.pt"))

        print("Evaluating val dataset:")
        loss_avg, val_result = evaluate(model, val_dataset)
        print("Loss average on val dataset: {}".format(loss_avg))
        if config.best_model_metric == "best_loss":
            if loss_avg.cpu().float() < best_loss:
                best_loss = loss_avg.cpu().float()
                print("Evaluating test dataset:")
                evaluate(model, test_dataset)
                torch.save(model.state_dict(), os.path.join(
                config.save_path, f"{config.fusion}_best.pt"
            ))
        else:
            if val_result[config.best_model_metric] > best_metric:
                best_metric = val_result[config.best_model_metric]
                print("Evaluating test dataset:")
                evaluate(model, test_dataset)
                torch.save(model.state_dict(), os.path.join(
                config.save_path, f"{config.fusion}_best.pt"
            ))
        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(
            config.save_path, f"{config.fusion}_best.pt"
        )))
            evaluate(model, test_dataset)
    if config.save_model:
        torch.save(model.state_dict(), os.path.join(config.save_path, f'final_model_{config.fusion}.pt'))



def evaluate(model, dataset):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
            model, dataset, config)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    result = ""
    key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
    for key in test_stats:
        if key in key_set:
            result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)   
    model.train()
    return loss_avg, test_stats



if __name__ == "__main__":
    config = parser.parse_args()
    load_args(YML_PATH[config.dataset], config)
    print(config)
    # set the seed value
    set_seed(config.seed)

    dataset_path = config.dataset_path

    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural')

    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural')

    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural')

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model = DFSP(config, attributes=attributes, classes=classes, offset=offset).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # if config.load_model is not False:
    #     model.load_state_dict(torch.load(config.load_model))

    os.makedirs(config.save_path, exist_ok=True)

    train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset)

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    print("done!")
