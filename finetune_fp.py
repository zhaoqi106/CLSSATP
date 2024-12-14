import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from utils.utils import get_logger

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.dataset_test import MolTestDatasetWrapper


#
# apex_support = False
# try:
#     sys.path.append('./apex')
#     from apex import amp
#
#     apex_support = True
# except:
#     print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
#     apex_support = False

fp_name = 'mixed'
downstream_task = 'fishdata'


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):
    def __init__(self, train_loader, valid_loader, test_loader, config, logger):
        self.logger = logger
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = 'fp_' + fp_name + '_' + downstream_task + '_' + current_time + '_' + config['task_name'] + '_' + config['dataset']['target']
        log_dir = os.path.join(self.config['dataset']['splitting'] + '/finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        logger = self.logger
        train_loader, valid_loader, test_loader = self.train_loader, self.valid_loader, self.test_loader

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for d in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        if self.config['model_type'] == 'gin':
            from models.ginet_finetune_fp import GINet
            model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        # elif self.config['model_type'] == 'gcn':
        #     # from models.gcn_finetune import GCN
        #     model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
        #     model = self._load_pre_trained_weights(model)

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_head' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        # if apex_support and self.config['fp16_precision']:
        #     model, optimizer = amp.initialize(
        #         model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
        #     )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0
        total_bn = len(train_loader)

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())
                    logger.info(
                        "task: {}, target: {}, type: {}, epoch: {}, bn / total_bn: {} / {}, train_loss : {}".format(
                            self.config['task_name'], self.config['dataset']['target'], config['dataset']['task'],
                            epoch_counter, bn, total_bn, loss.item()))

                # if apex_support and self.config['fp16_precision']:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification':
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression':
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                logger.info("task: {}, target: {}, type: {}, epoch: {}, valid_loss : {}".format(
                    self.config['task_name'], self.config['dataset']['target'], config['dataset']['task'],
                    epoch_counter, valid_loss))
                valid_n_iter += 1

        self._test(model, test_loader)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model_' + fp_name + '.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            logger.info("Loaded pre-trained model model_" + fp_name + ".pth with success.")
        except FileNotFoundError:
            logger.info("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data

        model.train()

        # self.logger.info("validation, num_data: {}, len(labels): {}".format(num_data, len(labels)))

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if len(labels) == 1:
                roc_auc = 100
                self.logger.info("len(data) == 1")
            else:
                roc_auc = roc_auc_score(labels, predictions[:, 1])

            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:

                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
                logger.info('task: {}, target: {}, Test loss: {}, Test MAE: {}'.format(self.config['task_name'],
                                                                                       self.config['dataset']['target'],
                                                                                       test_loss, self.mae))
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)
                logger.info('task: {}, target: {}, Test loss: {}, Test RMSE: {}'.format(self.config['task_name'],
                                                                                        self.config['dataset'][
                                                                                            'target'], test_loss,
                                                                                        self.rmse))

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if len(labels) == 1:
                self.roc_auc = -1
                logger.info("len(labels) == 1")
            else:
                self.roc_auc = roc_auc_score(labels, predictions[:, 1])
                true_label = labels
                predicted_label = predictions[:, 1].flatten()
                predicted_label = np.where(predicted_label >= 0.5, 1, 0)
                TP = np.sum((true_label == 1) & (predicted_label == 1))
                TN = np.sum((true_label == 0) & (predicted_label == 0))
                FP = np.sum((true_label == 0) & (predicted_label == 1))
                FN = np.sum((true_label == 1) & (predicted_label == 0))
                ACC = (TP + TN) / (TP + TN + FP + FN)
                RE = TP / (TP + FN) if (TP + FN) > 0 else 0
                PR = TP / (TP + FP) if (TP + FP) > 0 else 0
                self.acc = ACC
                self.re = RE
                self.pr = PR


            df = pd.DataFrame({
                'Label': labels,
                'Prediction': predictions[:, 1]
            })
            df.to_csv(
                self.config['dataset']['splitting'] + '/roc/fp_' + fp_name + '_' + downstream_task + '_{}_{}_finetune_{}_{}.csv'.format(config['fine_tune_from'], config['task_name'],
                                                                    config['dataset']['target'], self.roc_auc),
                mode='a', index=False,
            )
            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)
            self.logger.info('task: {}, target: {}, Test loss: {}, Test roc_auc: {}'.format(self.config['task_name'],
                                                                                       self.config['dataset']['target'],
                                                                                       test_loss, self.roc_auc))


def preprocess_data():
    data = pd.read_csv('./data/downstream_data/fish_data/model_data.csv')
    data = data[data['remove'].isnull()]
    data = data.rename(columns = {'SMILES' : 'smiles'})
    FHM_data = data[data['species'] == 'FHM']
    BS_data = data[data['species'] == 'BS']
    RT_data = data[data['species'] == 'RT']
    SHM_data = data[data['species'] == 'SHM']
    FHM_len = len(FHM_data)
    BS_len = len(BS_data)
    RT_len = len(RT_data)
    SHM_len = len(SHM_data)
    train_size = [int(0.8 * FHM_len), int(0.8 * BS_len), int(0.8 * RT_len), int(0.8 * SHM_len)]
    valid_size = [int(0.1 * FHM_len), int(0.1 * BS_len), int(0.1 * RT_len), int(0.1 * SHM_len)]
    test_size = [FHM_len - int(0.8 * FHM_len) - int(0.1 * FHM_len), BS_len - int(0.8 * BS_len) - int(0.1 * BS_len), RT_len - int(0.8 * RT_len) - int(0.1 * RT_len), SHM_len - int(0.8 * SHM_len) - int(0.1 * SHM_len)]


    for i in range(5):
        print("{} / 5".format(i + 1))
        random_seed = 42 + i
        shuffled_FHM = FHM_data.sample(frac=1).reset_index(drop=True)
        FHM_train = shuffled_FHM[:train_size[0]]
        FHM_val = shuffled_FHM[train_size[0]:train_size[0] + valid_size[0]]
        FHM_test = shuffled_FHM[train_size[0] + valid_size[0]:]

        shuffled_BS = BS_data.sample(frac=1).reset_index(drop=True)
        BS_train = shuffled_BS[:train_size[1]]
        BS_val = shuffled_BS[train_size[1]:train_size[1] + valid_size[1]]
        BS_test = shuffled_BS[train_size[1] + valid_size[1]:]

        shuffled_RT = RT_data.sample(frac=1).reset_index(drop=True)
        RT_train = shuffled_RT[:train_size[2]]
        RT_val = shuffled_RT[train_size[2]:train_size[2] + valid_size[2]]
        RT_test = shuffled_RT[train_size[2] + valid_size[2]:]

        shuffled_SHM = SHM_data.sample(frac=1).reset_index(drop=True)
        SHM_train = shuffled_SHM[:train_size[3]]
        SHM_val = shuffled_SHM[train_size[3]:train_size[3] + valid_size[3]]
        SHM_test = shuffled_SHM[train_size[3] + valid_size[3]:]

        total_train_set = pd.concat([FHM_train, BS_train, RT_train, SHM_train], ignore_index=True)
        total_train_set = total_train_set.groupby('smiles').agg({
            'uid': 'first',
            'CAS No.': 'first',
            'Toxicity (mg/L)': 'mean',
            'species': 'first',
            'cid': 'first',
            'Canonical_Smiles': 'first',
            'pLD50': 'mean',
            'remove': 'first',
            'pLD50-max': 'first',
            'pLD50-min': 'first',
            'pLD50-mean': 'first',
            'pLD50-median': 'first',
            'pLD50-final': 'first'
        }).reset_index()
        total_train_set['label'] = total_train_set['pLD50'].apply(lambda x:
                                                                  1 if x <= -1.0 else 0)
        total_train_set.to_csv('./data/downstream_data/fish_data/five_fold/total_train_fold_{}.csv'.format(i + 1), mode = 'w')

        FHM_val['label'] = FHM_val['pLD50'].apply(lambda x:
                                                  1 if x <= -1.0 else 0)
        FHM_test['label'] = FHM_test['pLD50'].apply(lambda x:
                                                    1 if x <= -1.0 else 0)
        FHM_val.to_csv('./data/downstream_data/fish_data/five_fold/FHM_val_fold_{}.csv'.format(i + 1), mode = 'w')
        FHM_test.to_csv('./data/downstream_data/fish_data/five_fold/FHM_test_fold_{}.csv'.format(i + 1), mode = 'w')

        BS_val['label'] = BS_val['pLD50'].apply(lambda x:
                                                  1 if x <= -1.0 else 0)
        BS_test['label'] = BS_test['pLD50'].apply(lambda x:
                                                    1 if x <= -1.0 else 0)
        BS_val.to_csv('./data/downstream_data/fish_data/five_fold/BS_val_fold_{}.csv'.format(i + 1), mode = 'w')
        BS_test.to_csv('./data/downstream_data/fish_data/five_fold/BS_test_fold_{}.csv'.format(i + 1), mode = 'w')

        RT_val['label'] = RT_val['pLD50'].apply(lambda x:
                                                  1 if x <= -1.0 else 0)
        RT_test['label'] = RT_test['pLD50'].apply(lambda x:
                                                    1 if x <= -1.0 else 0)
        RT_val.to_csv('./data/downstream_data/fish_data/five_fold/RT_val_fold_{}.csv'.format(i + 1), mode = 'w')
        RT_test.to_csv('./data/downstream_data/fish_data/five_fold/RT_test_fold_{}.csv'.format(i + 1), mode = 'w')

        SHM_val['label'] = SHM_val['pLD50'].apply(lambda x:
                                                  1 if x <= -1.0 else 0)
        SHM_test['label'] = SHM_test['pLD50'].apply(lambda x:
                                                    1 if x <= -1.0 else 0)
        SHM_val.to_csv('./data/downstream_data/fish_data/five_fold/SHM_val_fold_{}.csv'.format(i + 1), mode='w')
        SHM_test.to_csv('./data/downstream_data/fish_data/five_fold/SHM_test_fold_{}.csv'.format(i + 1), mode='w')


def main(config, logger):
    results = []
    results_acc = []
    results_re = []
    results_pr = []
    for i in range(5):
        train_dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
        train_dataset.set_data_path('./data/downstream_data/fish_data/five_fold/total_train_fold_{}.csv'.format(i + 1))
        valid_dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
        valid_dataset.set_data_path('./data/downstream_data/fish_data/five_fold/{}_val_fold_{}.csv'.format(config['task_name'], i + 1))
        test_dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
        test_dataset.set_data_path('./data/downstream_data/fish_data/five_fold/{}_test_fold_{}.csv'.format(config['task_name'], i + 1))
        # 防止只有某些集当中只有一类的情况发生
        if config['dataset']['task'] == 'classification':
            train_loader = train_dataset.get_fish_data_loaders()
            valid_loader = valid_dataset.get_fish_data_loaders()
            test_loader = test_dataset.get_fish_data_loaders()
            label1 = []
            label2 = []
            label3 = []
            for bn, data in enumerate(train_loader):
                print("check train_loader: {}/{}".format(bn + 1, len(train_loader)))
                label1.extend(data.y.flatten().numpy())
            label1 = set(label1)
            if len(label1) == 1:
                i -= 1
                logger.info("train set only contains one class!")
                continue
            for bn, data in enumerate(valid_loader):
                print("check valid_loader: {}/{}".format(bn + 1, len(valid_loader)))
                label2.extend(data.y.flatten().numpy())
            label2 = set(label2)
            if len(label2) == 1:
                i -= 1
                logger.info("valid set only contains one class!")
                continue
            for bn, data in enumerate(test_loader):
                print("check test_loader: {}/{}".format(bn + 1, len(test_loader)))
                label3.extend(data.y.flatten().numpy())
            label3 = set(label3)
            if len(label3) == 1:
                i -= 1
                logger.info("train set only contains one class!")
                continue
            logger.info("train set contains {} classes, valid set contains {} classes, test set contains {} classes".format(len(label1), len(label2), len(label3)))
        fine_tune = FineTune(train_loader, valid_loader, test_loader, config, logger)
        fine_tune.train()

        # Store the result based on the task type
        if config['dataset']['task'] == 'classification':
            results.append(fine_tune.roc_auc)
            results_acc.append(fine_tune.acc)
            results_pr.append(fine_tune.pr)
            results_re.append(fine_tune.re)
    # dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    # fine_tune = FineTune(dataset, config)
    # fine_tune.train()
    #
    # # Store the result based on the task type
    # if config['dataset']['task'] == 'classification':
    #     results.append(fine_tune.roc_auc)
    # if config['dataset']['task'] == 'regression':
    #     if config['task_name'] in ['qm7', 'qm8', 'qm9']:
    #         results.append(fine_tune.mae)
    #     else:
    #         results.append(fine_tune.mae)

    # Convert the list to a numpy array for easy computation of mean and std dev
    results = np.array(results)

    # Return mean and standard deviation of results
    return np.mean(results), np.mean(results_acc), np.mean(results_re), np.mean(results_pr)

if __name__ == "__main__":
    config = yaml.load(open("config_finetune_fp_" + fp_name + '_' + downstream_task + ".yaml", "r"), Loader=yaml.FullLoader)
    logger = get_logger("CLSSFP" + fp_name + '_' + downstream_task + "_finetune_" + config['dataset']['splitting'])
    logger.info(config)

    finish_bool = [0, 0, 0, 0]
    task_list = ['BS', 'FHM', 'RT', 'SHM']
    task_idx = {'BS': 0, 'FHM': 1, 'RT': 2, 'SHM': 3}

    preprocess_data()
    for task in task_list:
        config['task_name'] = task
        if config['task_name'] == 'BS':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/downstream_data/fish_data/BS_Origin.csv'
            target_list = ['label']
        elif config['task_name'] == 'FHM':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/downstream_data/fish_data/FHM_Origin.csv'
            target_list = ['label']
        elif config['task_name'] == 'RT':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/downstream_data/fish_data/RT_Origin.csv'
            target_list = ['label']
        elif config['task_name'] == 'SHM':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/downstream_data/fish_data/SHM_Origin.csv'
            target_list = ['label']
        else:
            raise ValueError('Undefined downstream task!')

        print(config)
        logger.info(
            "--------------------task:{}--------------------".format(config["task_name"]))

        results_list = []
        fin_score0 = 0
        fin_score1 = 0
        fin_score2 = 0
        fin_score3 = 0
        for target in target_list:
            config['dataset']['target'] = target
            auc, acc, re, pr = main(config, logger)
            results_list.append([target, auc, acc, re, pr])
            fin_score0 += auc
            fin_score1 += acc
            fin_score2 += re
            fin_score3 += pr

        fin_score0 /= len(target_list)
        fin_score1 /= len(target_list)
        fin_score2 /= len(target_list)
        fin_score3 /= len(target_list)

        if config['dataset']['splitting'] == 'scaffold':
            save_dir_name = 'scaffold'
        elif config['dataset']['splitting'] == 'random':
            save_dir_name = 'random'
        elif config['dataset']['splitting'] == 'random_scaffold':
            save_dir_name = 'random-scaffold'
        else:
            raise ValueError("splitting must be in random/scaffold/random_scaffold!")
        os.makedirs('./' + save_dir_name + '/experiments', exist_ok=True)
        df = pd.DataFrame(results_list)
        df.to_csv(
            save_dir_name + '/experiments/fp_' + fp_name + '_' + downstream_task + '_{}_{}_finetune_4_index.csv'.format(
                config['fine_tune_from'], config['task_name']),
            mode='a', index=False, header=False
        )