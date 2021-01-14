import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import OrderedDict

import nn.losses as losses_utils
import nn.tools as tools
import helpers


class Learner:
    def __init__(self, net, criterion, optimizer, scheduler=None, max_epochs=50):
        """
        The learner for carvana used for training and launching predictions
        Args:
            net (nn.Module): The neural net module containing the definition of your model
            criterion: 损失函数
            max_epochs (int): The maximum number of epochs on which the model will train
        """
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.epoch_counter = 0
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.net.to(self.device)

    def restore_model(self, model_path):
        """
            Restore a model parameters from the one given in argument
        Args:
            model_path (str): The path to the model to restore

        """
        self.net.load_state_dict(torch.load(model_path)).to(self.device)


    def _validate_epoch(self, valid_loader, threshold):
        """负责在一个epoch验证"""
        losses = tools.AverageMeter()
        dice_coeffs = tools.AverageMeter()

        it_count = len(valid_loader)
        batch_size = valid_loader.batch_size

        images = None  # To save the last images batch
        targets = None  # To save the last target batch
        preds = None  # To save the last prediction batch
        with tqdm(total=it_count, desc="Validating", leave=False) as pbar:
            for ind, (images, targets) in enumerate(valid_loader):
                if self.use_cuda:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                with torch.no_grad():
                    # forward
                    logits = self.net(images)
                    probs = torch.sigmoid(logits)
                    preds = (probs > threshold).float()

                    loss = self.criterion(logits, targets)
                    acc = losses_utils.dice_coeff(preds, targets)
                    losses.update(loss.data.item(), batch_size)
                    dice_coeffs.update(acc.data.item(), batch_size)
                pbar.update(1)
        batch_results = [ x.cpu() if self.use_cuda else x for x in [images[:2], targets[:2], preds[:2]]]
        return losses.avg, dice_coeffs.avg, batch_results


    def _train_epoch(self, train_loader, threshold):
        """负责训练一个epoch"""
        losses = tools.AverageMeter()  # 每个epoch都一个meter
        dice_coeffs = tools.AverageMeter()

        # Total training files count / batch_size
        batch_size = train_loader.batch_size
        it_count = len(train_loader)
        with tqdm(total=it_count,
                  desc=f"Epochs {self.epoch_counter + 1}/{self.max_epochs}",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]'
                  ) as pbar:
            for ind, (inputs, target) in enumerate(train_loader):

                inputs, target = inputs.contiguous(), target.contiguous()  # image=【8, 3, 572, 572】，mask=[8, 388, 388]了
                if self.use_cuda:
                    inputs = inputs.to(self.device)
                    target = target.to(self.device)
                
                # forward
                logits = self.net.forward(inputs) # [8, 388, 388]
                probs = torch.sigmoid(logits)
                pred = (probs > threshold).float()

                # backward + optimize
                loss = self.criterion(logits, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                
                # print statistics
                acc = losses_utils.dice_coeff(pred, target)

                losses.update(loss.data.item(), batch_size) # 将一个batch的平均loss输入
                dice_coeffs.update(acc.data.item(), batch_size) # 一个batch的平均准确率

                # Update pbar
                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(loss.data.item()),
                                             dice_coeff='{0:1.5f}'.format(acc.data.item()))) # 在最后显示该batch的损失和准确率
                pbar.update(1) # 进度条涨1
                
        return losses.avg, dice_coeffs.avg # 返回的是当前epoch所有样本(这里是像素点)的平均损失和精确度

    @helpers.st_time(show_func_name=False)
    def _run_epoch(self, train_loader, valid_loader, threshold=0.5, callbacks=None):   
        """负责在一个epoch的训练集上训练、验证集上验证、回调"""

        self.net.train()
        train_loss, train_dice_coeff = self._train_epoch(train_loader, threshold)

        self.net.eval()
        val_loss, val_dice_coeff, last_batch_outs = self._validate_epoch(valid_loader, threshold)

        # If there are callback call their __call__ method and pass in some arguments
        if callbacks:
            for cb in callbacks: 
                cb(step_name="epoch",
                   net=self.net,
                   last_val_batch=last_batch_outs,
                   epoch_id=self.epoch_counter + 1,
                   train_loss=train_loss, train_dice_coeff=train_dice_coeff,
                   val_loss=val_loss, val_dice_coeff=val_dice_coeff,
                   )
        print("train_loss = {:03f}, train_dice_coeff = {:03f}\n"
              "val_loss   = {:03f}, val_dice_coeff   = {:03f}"
              .format(train_loss, train_dice_coeff, val_loss, val_dice_coeff))
        self.epoch_counter += 1


    def train(self, train_loader: DataLoader, valid_loader: DataLoader, threshold=0.5, callbacks=None):
        """
            负责跑多个epoch进行网络训练
        Args:
            train_loader (DataLoader): The Dataloader for training
            valid_loader (DataLoader): The Dataloader for validation
            threshold (float): The threshold used to consider the mask present or not
            callbacks (list): List of callbacks functions to call at each epoch
        Returns:
            str, None: The path where the model was saved, or None if it wasn't saved
        """
        for epoch in range(self.max_epochs):
            self._run_epoch(train_loader, valid_loader, threshold, callbacks)

        # 全部epoch跑完后，调用保存模型的callback回调函数。
        if callbacks:
            for cb in callbacks:  # 这里callback根据step_name，相当于只使用save模型的那个callback。
                cb(step_name="train", net=self.net, epoch_id=self.epoch_counter+1,)
        print("Training finished!")


    def predict(self, test_loader, callbacks=None):
        """
            Launch the prediction on the given loader and pass
            each predictions to the given callbacks.
        Args:
            test_loader (DataLoader): The loader containing the test dataset
            callbacks (list): List of callbacks functions to call at prediction pass
        """
        # Switch to evaluation mode
        self.net.eval()

        it_count = len(test_loader)

        with tqdm(total=it_count, desc="Classifying") as pbar:
            for ind, (images, files_name) in enumerate(test_loader):
                if self.use_cuda:
                    images = images.to(self.device)

                with torch.no_grad():
                    # forward
                    logits = self.net(images)
                    probs = torch.sigmoid(logits)
                    probs = probs.data.cpu().numpy() if self.use_cuda else probs.data.numpy()

                # If there are callback call their __call__ method and pass in some arguments
                if callbacks:
                    for cb in callbacks:
                        cb(step_name="predict",
                           net=self.net,
                           probs=probs,
                           files_name=files_name
                           )

                pbar.update(1)

        print("Predict finished!")
