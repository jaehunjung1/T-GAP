import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import tensorboard
import dgl
import numpy as np
from tqdm import tqdm

from args import get_args, set_properties_to_args
from dataset import get_datasets
from model import TGAP
from util import *
import ipdb


def train_one_epoch(model, dataloader, optimizer, args, epoch, global_count, writer):
    model.train()
    epoch_count = 0
    epoch_loss = 0.
    epoch_correct1, epoch_correct3, epoch_correct10 = 0., 0., 0.

    with tqdm(dataloader, desc=f"Train Ep {epoch}", mininterval=60) as tq:
        for batch in tq:
            batch["head"] = batch["head"].to(args.device)
            batch["relation"] = batch["relation"].to(args.device)
            batch["tail"] = batch["tail"].to(args.device)
            batch["time"] = batch["time"].to(args.device)
            batch["graph"].to(args.device)

            attention_history = model(batch)
            predicted_prob = attention_history[-1].transpose(0, 1)

            # Compute loss
            loss = F.nll_loss(torch.log(predicted_prob + 1e-12), batch["tail"])

            if args.dataset == 'data/wikidata11k_aug':
                if epoch_count % 16 == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                loss = loss.item()

            if loss != loss:
                print(f"Nan came up at epoch {epoch}")
                ipdb.set_trace()

            epoch_loss += loss * batch["head"].size(0)
            epoch_count += batch["head"].size(0)
            avg_loss = epoch_loss / epoch_count

            # Compute hits@k
            epoch_correct1 += hits_at_k(predicted_prob, batch["tail"], k=1)
            epoch_correct3 += hits_at_k(predicted_prob, batch["tail"], k=3)
            epoch_correct10 += hits_at_k(predicted_prob, batch["tail"], k=10)

            avg_hits1 = epoch_correct1 / epoch_count
            avg_hits3 = epoch_correct3 / epoch_count
            avg_hits10 = epoch_correct10 / epoch_count

            if args.dataset == 'data/wikidata11k_aug':
                if epoch_count % 16 == 0:
                    tq.set_postfix({'Avg loss': avg_loss}, refresh=False)
                    writer.add_scalar('Loss/Train_Avg_Loss', avg_loss,
                                      global_step=global_count+epoch_count)
                    writer.add_scalar('Metric/Train_hits@1', avg_hits1,
                                      global_step=global_count+epoch_count)
                    writer.add_scalar('Metric/Train_hits@3', avg_hits3,
                                      global_step=global_count+epoch_count)
                    writer.add_scalar('Metric/Train_hits@10', avg_hits10,
                                      global_step=global_count + epoch_count)
            else:
                tq.set_postfix({'Avg loss': avg_loss}, refresh=False)
                writer.add_scalar('Loss/Train_Avg_Loss', avg_loss,
                                  global_step=global_count + epoch_count)
                writer.add_scalar('Metric/Train_hits@1', avg_hits1,
                                  global_step=global_count + epoch_count)
                writer.add_scalar('Metric/Train_hits@3', avg_hits3,
                                  global_step=global_count + epoch_count)
                writer.add_scalar('Metric/Train_hits@10', avg_hits10,
                                  global_step=global_count + epoch_count)

    return epoch_count


def evaluate(model, dataloader, args, epoch, writer=None, mode='Valid'):
    model.eval()
    total_count = 0
    total_loss = 0.
    total_correct1, total_correct3, total_correct10 = 0., 0., 0.
    mrr = 0.

    with tqdm(dataloader, desc=mode, mininterval=5) as tq:
        for batch in tq:
            batch["head"] = batch["head"].to(args.device)
            batch["relation"] = batch["relation"].to(args.device)
            batch["tail"] = batch["tail"].to(args.device)
            batch["time"] = batch["time"].to(args.device)
            batch["graph"].to(args.device)

            with torch.no_grad():
                attention_history = model(batch)
                predicted_prob = attention_history[-1].detach().transpose(0, 1)

            # Compute loss
            loss = F.nll_loss(torch.log(predicted_prob + 1e-12), batch["tail"])
            loss = loss.item()
            total_loss += loss * batch["head"].size(0)

            # Compute hits@k
            total_correct1 += hits_at_k(predicted_prob, batch["tail"], k=1)
            total_correct3 += hits_at_k(predicted_prob, batch["tail"], k=3)
            total_correct10 += hits_at_k(predicted_prob, batch["tail"], k=10)
            total_count += batch["head"].size(0)

            # Compute MRR
            sorted_prob = torch.argsort(predicted_prob, dim=-1, descending=True)
            ranks = torch.tensor([sorted_prob[i].eq(batch['tail'][i]).nonzero().item()
                                  for i in range(len(batch['tail']))])
            mrr += torch.sum(torch.reciprocal(ranks.float() + 1))

            tq.set_postfix({'Avg hits@1': total_correct1 / total_count})

    avg_loss = total_loss / total_count
    avg_hits1 = total_correct1 / total_count
    avg_hits3 = total_correct3 / total_count
    avg_hits10 = total_correct10 / total_count
    mrr = mrr / total_count

    if writer is not None:
        writer.add_scalar(f"Loss/{mode}_Loss", avg_loss, global_step=epoch)
        writer.add_scalar(f"Metric/{mode}_hits@1", avg_hits1, global_step=epoch)
        writer.add_scalar(f"Metric/{mode}_hits@3", avg_hits3, global_step=epoch)
        writer.add_scalar(f"Metric/{mode}_hits@10", avg_hits10, global_step=epoch)

    return avg_loss, avg_hits1, avg_hits3, avg_hits10, mrr


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dgl.random.seed(args.seed)

    filenames = [args.train_fname, args.valid_fname, args.test_fname]

    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Configure dataset and dataloader
    train_dataset, valid_dataset, test_dataset = get_datasets(filenames, args.device)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=train_dataset.collate, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=valid_dataset.collate, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=valid_dataset.collate, num_workers=4, pin_memory=True)

    args = set_properties_to_args(args, train_dataset.kg.entity_vocab,
                                  train_dataset.kg.relation_vocab, train_dataset.kg.time_vocab)

    # Configure model, optimizer, scheduler
    model = TGAP(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=args.patience)

    if args.test:
        assert os.path.exists(args.ckpt), "Checkpoint file does not exist."
        load_checkpoint(args.ckpt, model, optimizer, scheduler)

        test_loss, test_hits1, test_hits3, test_hits10, mrr = evaluate(model, test_dataloader, args, 1, mode="Test")
        print(f"Test Loss: {test_loss:.5} \n"
              f"MRR: {mrr:.5} \n"
              f"Hits@1: {test_hits1:.5} \n"
              f"Hits@3: {test_hits3:.5} \n"
              f"Hits@10: {test_hits10:.5}")

    else:
        best_valid_hits1 = -1.
        global_count = 0
        summary_writer = tensorboard.SummaryWriter(log_dir=args.tensorboard_dir)
        summary_writer.add_text("Args", str(args), 0)

        if args.ckpt:
            load_checkpoint(args.ckpt, model, optimizer, scheduler)

        for epoch in range(1, args.epoch + 1):
            global_count += train_one_epoch(model, train_dataloader, optimizer, args,
                                            epoch, global_count, summary_writer)

            valid_loss, valid_hits1, _, _, _ = evaluate(model, valid_dataloader, args, epoch, summary_writer)
            scheduler.step(valid_loss)

            if best_valid_hits1 < valid_hits1:
                best_valid_hits1 = valid_hits1
                save_checkpoint(args.ckpt_dir, model, optimizer, scheduler, epoch, global_count, valid_hits1)


if __name__ == "__main__":
    args = get_args()
    main(args)
