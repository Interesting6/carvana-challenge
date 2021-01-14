import nn.learner
import nn.unet_origin as unet_origin
import nn.unet as unet_custom
from nn.losses import customLoss
from nn.train_callbacks import TensorboardVisualizerCallback, TensorboardLoggerCallback, ModelSaverCallback
from nn.test_callbacks import PredictionsSaverCallback
import helpers

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.optim as optim

import img.augmentation as aug
from data.fetcher import DatasetFetcher


import os
from multiprocessing import cpu_count

from data.dataset import TrainImageDataset, TestImageDataset
import multiprocessing


def main():
    # Hyperparameters
    input_img_resize = (572, 572)  # The resize size of the input images of the neural net
    output_img_resize = (388, 388)  # The resize size of the output images of the neural net
    # input_img_resize = (1024, 1024)
    # output_img_resize = (1024, 1024)
    batch_size = 8
    epochs = 50
    threshold = 0.5
    validation_size = 0.2
    sample_size = 0.05 # None  # Put 'None'为选择全部数据，否者0-1之间为随机抽取部分数据
    now_time = helpers.get_model_timestamp()

    # -- Optional parameters 一些有用的参数和callback回调函数
    threads = 0 #cpu_count()
    use_cuda = torch.cuda.is_available()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Training callbacks
    tb_viz_cb = TensorboardVisualizerCallback(os.path.join(script_dir, '../logs2/tb_viz'))
    tb_logs_cb = TensorboardLoggerCallback(os.path.join(script_dir, '../logs2/tb_logs'))
    model_saver_cb = ModelSaverCallback(os.path.join(script_dir, f'../output2/models/model_{now_time}'), verbose=True)

    # Download the datasets
    ds_fetcher = DatasetFetcher()
    ds_fetcher.download_dataset()

    # Get the path to the files for the neural net
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(sample_size=sample_size,
                                                                    validation_size=validation_size)
    full_x_test = ds_fetcher.get_test_files(sample_size)  # 取出测试集

    # -- Computed parameters
    # Get the original images size (assuming they are all the same size)
    origin_img_size = ds_fetcher.get_image_size(X_train[0]) # 【1918，1280】

    # Testing callbacks
    pred_saver_cb = PredictionsSaverCallback(os.path.join(script_dir, '../output2/submit.csv.gz'),
                                             origin_img_size, threshold)

    # -- Define our neural net architecture
    #net = unet_custom.UNet1024((3, *input_img_resize))
    net = unet_origin.UNetOriginal((3, *input_img_resize))
    criterion = customLoss()
    # optimizer = optim.RMSprop(net.parameters(), lr=0.0002)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.99)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    learner = nn.learner.Learner(net, criterion, optimizer, scheduler, epochs) # 与torch.nn无关

    train_ds = TrainImageDataset(X_train, y_train, input_img_resize, output_img_resize,
                                 X_transform=aug.augment_img) # 只对image增强，mask也对应的变换；但不对mask增强
    train_loader = DataLoader(train_ds, batch_size,
                              sampler=RandomSampler(train_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    valid_ds = TrainImageDataset(X_valid, y_valid, input_img_resize, output_img_resize,
                                 threshold=threshold)
    valid_loader = DataLoader(valid_ds, batch_size,
                              sampler=SequentialSampler(valid_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    print("Training on {} samples and validating on {} samples "
          .format(len(train_loader.dataset), len(valid_loader.dataset)))

    # Train the learner
    learner.train(train_loader, valid_loader, callbacks=[tb_viz_cb, tb_logs_cb, model_saver_cb])

    test_ds = TestImageDataset(full_x_test, input_img_resize)
    test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

    # Predict & save
    learner.predict(test_loader, callbacks=[pred_saver_cb])
    pred_saver_cb.close_saver()


if __name__ == "__main__":
    # Workaround for a deadlock issue on Pytorch 0.2.0: https://github.com/pytorch/pytorch/issues/1838
    # multiprocessing.set_start_method('spawn', force=True)
     main()
