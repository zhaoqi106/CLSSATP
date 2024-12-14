import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.nt_xent import NTXentLoss
from utils.utils import get_logger


# from utils.nt_xent import NTXentLoss
# from apex import amp

# apex_support = False
# try:
#     sys.path.append('./apex')
#     from apex import amp
#
#     apex_support = True
# except:
#     print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
#     apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class CLSSATP(object):
    def __init__(self, dataset, config, logger):
        self.logger = logger
        self.config = config
        self.device = self._get_device()

        dir_name = datetime.now().strftime('%b%d_%H-%M-%S_fpmodel_mixed')
        log_dir = os.path.join('ckpt', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        self.class_criterion = torch.nn.CrossEntropyLoss()

        """
        pos_arr = np.load('./temp/pos_arr.npy', allow_pickle=True)
        neg_arr = np.load('./temp/neg_arr.npy', allow_pickle=True)

        pos_weight = torch.tensor([neg_arr[i] / pos_arr[i] for i in range(len(pos_arr))])
        """

        self.fp_class_criterion = torch.nn.BCEWithLogitsLoss()

        self.dataset = dataset
        # self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, data_i, n_iter):
        x, pred_maccs, pred_morgan, pred_pubchem = model(data)

        pre_maccs_label = pred_maccs.float()
        maccs_fp = data.fp_maccs.view(self.config['batch_size'], -1).float()
        maccs_loss = self.fp_class_criterion(pre_maccs_label, maccs_fp)
        maccs_loss = maccs_loss.mean()

        pre_morgan_label = pred_morgan.float()
        morgan_fp = data.fp_morgan.view(self.config['batch_size'], -1).float()
        morgan_loss = self.fp_class_criterion(pre_morgan_label, morgan_fp)
        morgan_loss = morgan_loss.mean()

        pre_pubchem_label = pred_pubchem.float()
        pubchem_fp = data.fp_pubchem.view(self.config['batch_size'], -1).float()
        pubchem_loss = self.fp_class_criterion(pre_pubchem_label, pubchem_fp)
        pubchem_loss = pubchem_loss.mean()

        x1, _, _, _ = model(data_i)
        x = F.normalize(x, dim=1)
        x1 = F.normalize(x1, dim=1)
        mask_loss = self.nt_xent_criterion(x, x1)
        loss = mask_loss + (maccs_loss + morgan_loss + pubchem_loss) / 3
        # loss = self.criterion(pred, data.y.flatten())
        return loss

    def train(self):
        logger = self.logger
        train_loader, valid_loader = self.dataset.get_data_loaders()
        logger.info("train_loader_bn: {}, batch_size: {}".format(len(train_loader), self.config['batch_size']))
        logger.info("valid_loader_bn: {}, batch_size: {}".format(len(valid_loader), self.config['batch_size']))

        if self.config['model_type'] == 'gin':
            from models.ginet_clssatp_mixed_fp import GINet
            model = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        else:
            raise ValueError('Undefined GNN model.')
        print(model)

        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'],
            weight_decay=eval(self.config['weight_decay'])
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs'] - self.config['warm_up'],
            eta_min=0, last_epoch=-1
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

        total_bn = len(train_loader)

        for epoch_counter in range(self.config['epochs']):
            for bn, (data, data_i) in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                data_i = data_i.to(self.device)
                loss = self._step(model, data, data_i, n_iter)

                # if n_iter % self.config['log_every_n_steps'] == 0:
                if bn % 150 == 0 or bn == total_bn - 1:
                    logger.info(
                        'epoch: {}, bn / total_bn: {} / {}, train_loss: {}'.format(epoch_counter, bn, total_bn, loss))
                    # if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                print(epoch_counter, f'{bn} / {total_bn}', loss.item())

                # if apex_support and self.config['fp16_precision']:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                #     loss.backward()
                loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                print(epoch_counter, bn, valid_loss, '(validation)')
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                logger.info("epoch: {}, validation_loss: {}".format(epoch_counter, valid_loss))
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            if (epoch_counter + 1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(),
                           os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))

            # warmup for the first few epochs
            if epoch_counter >= self.config['warm_up']:
                scheduler.step()

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (data, data_i) in valid_loader:
                data = data.to(self.device)
                data_i = data_i.to(self.device)

                loss = self._step(model, data, data_i, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter

        model.train()
        return valid_loss


def main():
    config = yaml.load(open("config_fp_mixed.yaml", "r"), Loader=yaml.FullLoader)
    logger = get_logger("CLSSFPmixedMol")
    print(config)
    logger.info(config)

    if config['aug'] == 'subgraph':
        from dataset.dataset_subgraph_mix_fp import MoleculeDatasetWrapper
    # elif config['aug'] == 'node':
    # from dataset.dataset import MoleculeDatasetWrapper
    # elif config['aug'] == 'mix':
    #     from dataset.dataset_mix import MoleculeDsatasetWrapper
    else:
        raise ValueError('Not defined molecule augmentation!')

    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    clssatp = CLSSATP(dataset, config, logger)
    clssatp.train()


if __name__ == "__main__":
    main()