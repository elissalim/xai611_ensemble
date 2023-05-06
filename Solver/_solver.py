import os
import numpy as np
import torch
from metrics import cal_log
from utils import print_update, createFolder, write_json, print_dict
from openTSNE import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
torch.set_printoptions(linewidth=1000)


class Solver:
    def __init__(self, args, net, train_loader, val_loader, criterion, optimizer, scheduler, log_dict):
        self.args = args
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_dict = log_dict

    def train(self):
        log_tmp = {key: [] for key in self.log_dict.keys() if "train" in key}
        self.net.train()

        for i, data in enumerate(self.train_loader):
            # Load batch data
            inputs, labels = data[0].cuda(), data[1].cuda()

            # Feed-forward
            self.optimizer.zero_grad()
            if self.args.net == "EEGNet":
                outputs = self.net(inputs)
            elif self.args.net == "deep4":
                outputs = self.net(inputs.permute(0, 2, 3, 1))  # (B, 1, T, C)

            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Calculate log
            cal_log(log_tmp, outputs=outputs, labels=labels, loss=loss)

            # Print
            sentence = f"({(i + 1) * self.args.batch_size} / {len(self.train_loader.dataset.X)})"
            for key, value in log_tmp.items():
                sentence += f" {key}: {value[i]:0.3f}"
            print_update(sentence, i)
        print("")
        # Record log
        for key in log_tmp.keys():
            self.log_dict[key].append(np.mean(log_tmp[key]))

    def val(self, test=False):
        log_tmp = {key: [] for key in self.log_dict.keys() if "val" in key}
        self.net.eval()

        out_all, label_all, probs_all = [], [], []
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                # Load batch data
                inputs, labels = data[0].cuda(), data[1].cuda()

                # Feed-forward
                if self.args.net == "EEGNet":
                    outputs = self.net(inputs)
                elif self.args.net == "deep4":
                    outputs = self.net(inputs.permute(0, 2, 3, 1))  # (B, 1, T, C)

                if self.args.mode=='test':
                    if self.args.net == "EEGNet":
                        probs = self.get_prob(inputs)
                    elif self.args.net == "deep4":
                        probs = self.get_prob(inputs.permute(0, 2, 3, 1))  # (B, 1, T, C)
                    probs_all.append(probs.cpu().numpy())

                out_all.append(outputs.cpu().numpy()), label_all.append(labels.cpu().numpy())
                loss = self.criterion(outputs, labels)

                # Calculate log
                cal_log(log_tmp, outputs=outputs, labels=labels, loss=loss)

            # Record log
            for key in log_tmp.keys():
                self.log_dict[key].append(np.mean(log_tmp[key]))
        if test:
            out_all = np.concatenate(out_all, axis=0)
            label_all = np.concatenate(label_all, axis=0)

            # TSNE
            tsne_x = TSNE().fit(out_all)

            # plot
            plt.figure(figsize=(6, 5), dpi=300)
            scatter = plt.scatter(tsne_x[:, 0], tsne_x[:, 1], c=label_all, cmap='jet', s=10)
            plt.xticks([])
            plt.yticks([])
            plt.legend(*scatter.legend_elements(), loc='upper right')
            plt.tight_layout()
            # plt.savefig(f"./result/{self.args.stamp}/{self.args.train_subject[0]}_tsne.png", dpi=300)
            plt.savefig(f"{self.args.save_path[:-1]}{self.args.train_subject[0]}_tsne.png", dpi=300)
            # plt.show()

        if self.args.mode=='test':
            return np.concatenate(probs_all, axis=0)

    def experiment(self):
        print("[Start experiment]")
        total_epoch = self.args.epochs

        for epoch in range(1, total_epoch + 1):
            print(f"Epoch {epoch}/{total_epoch}")
            # Train
            self.train()

            # Validation
            self.val()

            # Print
            print("=>", end=' ')
            for key, value in self.log_dict.items():
                print(f"{key}: {value[epoch - 1]:0.3f}", end=' ')
            print("")
            print(f"=> learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")

            # Update scheduler
            self.scheduler.step() if self.scheduler else None

            # Save checkpoint
            createFolder(os.path.join(self.args.save_path, "checkpoint"))
            if epoch % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'net_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
                }, os.path.join(self.args.save_path, f"checkpoint/{epoch}.tar"))
                write_json(os.path.join(self.args.save_path, "log_dict.json"), self.log_dict)

        # Save args & log_dict
        self.args.seed = torch.initial_seed()
        self.args.cuda_seed = torch.cuda.initial_seed()
        self.args.acc = np.round(self.log_dict['val_acc'][-1], 3)
        delattr(self.args, 'topo') if hasattr(self.args, 'topo') else None
        delattr(self.args, 'phase') if hasattr(self.args, 'phase') else None
        write_json(os.path.join(self.args.save_path, "args.json"), vars(self.args))

        print("====================================Finish====================================")
        print(self.net, '\n')
        print_dict(vars(self.args))
        print(f"Last checkpoint: {os.path.join(self.args.save_path, 'checkpoint', str(epoch) + '.tar')}")

        return self.test()

    # TODO: change how test should be really done, split train-val-test
    def test(self):
        print("[Start test]")
        for epoch in range(1, 2):

            # Validation
            probs = self.val(test=True)

            # Print
            print("=>", end=' ')
            print(f"test acc: {self.log_dict['val_acc']}")
        print("====================================Finish====================================")

        return probs

    def get_prob(self, inputs):
        # pass the input through the model
        logits = self.net(inputs)

        # apply the softmax function to get the probability of each class prediction
        return F.softmax(logits, dim=1)