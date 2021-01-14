import cv2
import torch
import numpy as np
import scipy.misc as scipy
from tensorboardX import SummaryWriter
import copy


class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class TensorboardVisualizerCallback(Callback):
    def __init__(self, path_to_files):
        """
            Callback intended to be executed at each epoch
            of the training which goal is to display the result
            of the last validation batch in Tensorboard
        Args:
            path_to_files (str): The path where to store the log files
        """
        self.path_to_files = path_to_files

    def _apply_mask_overlay(self, image, mask, color=(0, 255, 0)): 
        """将mask用G绿色涂到image上"""
        mask = np.dstack((mask, mask, mask)) * np.array(color)  # 将mask转为RGB-3通道，并只保留G通道，其他两通道置0.
        mask = mask.astype(np.uint8)
        return cv2.addWeighted(mask, 0.5, image, 0.5, 0.)  # image * α + mask * β + λ  将mask叠加到image上。

    def _get_mask_representation(self, image, mask):
        """ 将mask涂到image上
         Given a mask and an image this method returns
         one image representing 3 patches of the same image.
         These patches represent:
            - The original image 原图
            - The original mask  原掩码
            - The mask applied to the original image 加了掩码的图
        Args:
            image (np.ndarray): The original image
            mask (np.ndarray): The predicted mask

        Returns (np.ndarray):
            An image of size (original_image_height, (original_image_width * 3))
            showing 3 patches of the original image
        """

        H, W, C = image.shape            
        mask = cv2.resize(mask, (H, W))  # 将mask resize为image同样的大小
        masked_img = self._apply_mask_overlay(image, mask) # 将真实mask涂到image上，*0.5所以图像变暗了

        results = np.zeros((H, 3 * W, 3), np.uint8)  # 初始化横着放的三个图：image、蓝mask黑底、绿色mask标注的image

        p = np.zeros((H * W, 3), np.uint8)
        m = np.zeros((H * W), np.uint8)
        l = mask.reshape(-1)

        a = (2 * l + m)  # m为全0，为啥要加m？    
        miss = np.where(a == 2)[0]   # missing -- false negative？预测为背景实际为前景，
        hit = np.where(a == 3)[0]    # 预测正确的？
        fp = np.where(a == 1)[0]     # false positive？预测错误？
        p[miss] = np.array([0, 0, 255]) # 蓝   
        p[hit] = np.array([64, 64, 64]) # 灰？
        p[fp] = np.array([0, 255, 0])   # 绿
        p = p.reshape(H, W, 3)

        results[:, 0:W] = image            # 原图image
        results[:, W: 2*W] = p             # 蓝色mask黑底
        results[:, 2*W: 3*W] = masked_img  # 绿色mask标注的image
        return results

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "epoch":
            return

        epoch_id = kwargs['epoch_id']
        last_images, last_targets, last_preds = kwargs['last_val_batch']  # 验证集的最后一个batch [8, 3, 572,572] 真实和预测标签[8, 388,388]
        writer = SummaryWriter(self.path_to_files)

        for i, (image, target_mask, pred_mask) in enumerate(zip(last_images, last_targets, last_preds)): # 对这个batch中的每张图

            image = image.data.float().numpy().astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))  # Invert [c,h,w] to [h,w,c]
            target_mask = target_mask.float().data.numpy().astype(np.uint8)
            pred_mask = pred_mask.float().data.numpy().astype(np.uint8)
            expected_result = self._get_mask_representation(image, target_mask) # 得到一个(image, 真mask, 真mask涂好image)的三图组合
            pred_result = self._get_mask_representation(image, pred_mask)       # 得到一个(image, 预测mask, 预测mask涂好image)的三图组合
            writer.add_image("Epoch_" + str(epoch_id) + '-Image_' + str(i + 1) + '-Expected', expected_result, epoch_id)
            writer.add_image("Epoch_" + str(epoch_id) + '-Image_' + str(i + 1) + '-Predicted', pred_result, epoch_id)
            if i == 1:  # 2 Images are sufficient，只把该batch中的第一张图和第二张图及其对应结果展示一下即ok
                break
        writer.close()


class TensorboardLoggerCallback(Callback):
    def __init__(self, path_to_files):
        """
            Callback intended to be executed at each epoch
            of the training which goal is to add valuable
            information to the tensorboard logs such as the losses
            and accuracies
        Args:
            path_to_files (str): The path where to store the log files
        """
        self.path_to_files = path_to_files

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "epoch":
            return

        epoch_id = kwargs['epoch_id']

        writer = SummaryWriter(self.path_to_files)
        writer.add_scalar('data/train_loss', kwargs['train_loss'], epoch_id)
        writer.add_scalar('data/train_dice_coeff', kwargs['train_dice_coeff'], epoch_id)
        writer.add_scalar('data/val_loss', kwargs['val_loss'], epoch_id)
        writer.add_scalar('data/val_dice_coeff', kwargs['val_dice_coeff'], epoch_id)
        writer.close()


class ModelSaverCallback(Callback):
    def __init__(self, path_to_model, verbose=False):
        """
            Callback intended to be executed each time a whole train pass
            get finished. This callback saves the model in the given path
            回调函数在每次完成整改训练时执行，此回调将模型保存在给定路径
        Args:
            verbose (bool): True or False to make the callback verbose
            path_to_model (str): The path where to store the model
        """
        self.verbose = verbose
        self.path_to_model = path_to_model
        self.suffix = ""
        self.best_loss = np.inf
        self.best_acc = 0.
        self.best_loss_weight = None
        self.best_acc_weight = None
        self.best_loss_epoch = None
        self.best_acc_epoch = None

    def set_suffix(self, suffix):
        """

        Args:
            suffix (str): The suffix to append to the model file name
        """
        self.suffix = suffix

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] not in ["train", "epoch"]:
            return

        if kwargs['step_name'] == "epoch": # 代表一个epoch的train和val结束，记录信息
            self._epoch_info_update(**kwargs)

        if kwargs['step_name'] == "train": # train函数结束，即代表所有epoch的训练结束
            self.set_suffix("train_end")
            # save last epoch weight
            pth = self.path_to_model + self.suffix + "_last_epoch.pt"
            net = kwargs['net']
            torch.save(net.state_dict(), pth)

            # save best loss & acc weight
            if self.best_acc_epoch:
                pth = self.path_to_model + self.loss_suffix
                torch.save(self.best_loss_weight, pth)
            if self.best_loss_epoch:
                pth = self.path_to_model + self.acc_suffix
                torch.save(self.best_acc_weight, pth)

            if self.verbose:
                print("Model saved in {}".format(pth))


    def _epoch_info_update(self, **kwargs):
        """
            只更新最好loss和acc的网络权重信息。
        """
        epoch_id = kwargs["epoch_id"]
        if kwargs["val_loss"] < self.best_loss:
            self.best_loss_weight = copy.deepcopy(kwargs['net'].state_dict())
            self.best_loss = kwargs["val_loss"]
            self.best_loss_epoch = epoch_id
            self.loss_suffix = f"best_loss_epoch_{epoch_id}.pt"
        if kwargs["val_dice_coeff"] > self.best_acc:
            self.best_acc_weight = copy.deepcopy(kwargs['net'].state_dict())
            self.best_acc = kwargs["val_dice_coeff"]
            self.best_acc_epoch = epoch_id
            self.acc_suffix = f"best_acc_epoch_{epoch_id}.pt"
    