import argparse
import os
import time
import gc
import datetime
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.modules.loss import _Loss
from torch.optim import lr_scheduler

from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.autograd import  Function
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CasualUNet_Dataset
import model
from model import CasualUNet
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子

class SingleSrcNegSDR(_Loss):
    r"""Base class for single-source negative SI-SDR, SD-SDR and SNR.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target and
            estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.
        reduction (string, optional): Specifies the reduction to apply to
            the output:
            ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.

    Shape:
        - est_targets : :math:`(batch, time)`.
        - targets: :math:`(batch, time)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch)` if ``reduction='none'`` else
        [] scalar if ``reduction='mean'``.

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(SingleSrcNegSDR("sisdr"),
        >>>                            pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, reduction="none", EPS=1e-8):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, est_target, target):
        if target.size() != est_target.size() or target.ndim != 2:
            raise TypeError(
                f"Inputs must be of shape [batch, time], got {target.size()} and {est_target.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(target, dim=1, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(est_target * target, dim=1, keepdim=True)
            # [batch, 1]
            s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + self.EPS
            # [batch, time]
            scaled_target = dot * target / s_target_energy
        else:
            # [batch, time]
            scaled_target = target
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = est_target - target
        else:
            e_noise = est_target - scaled_target
        # [batch]
        losses = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + self.EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses

class my_Loss(nn.Module):
    def __init__(self) -> None:
        super(my_Loss,self).__init__()
    def forward(self,target_speech,target_noise,input_speech):
        input_noise=target_speech+target_noise-input_speech
        return F.l1_loss(input_speech,target_speech)+F.l1_loss(input_noise,target_noise)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" #可用gpu列表
    device_ids=[4,5,6,7] #先尝试4卡
    device=device_ids[0]
    masktype='C'
    ##选定哪一种masktype
    GPU_NUMS=4 #4卡
    #set argument
    # args=mySetup()
    setup_seed(20)
    #设置随机数种子
    BATCH_SIZE=4*GPU_NUMS
    if not os.path.isdir('./tensorboard'):
        os.mkdir('./tensorboard')
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    NUM_EPOCHS=100
    LR=1e-3
    ADAM_BETA=(0.5,0.999)
    NUM_WORKERS=8
    IF_RESUME_BREAK=False
    NUM_MICS=torch.zeros(BATCH_SIZE).type(torch.FloatTensor) #这个地方设置成这样比较好，设置成1会导致无法多卡
    #add tensorboard path
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    tensorboard_path='./tensorboard/'+time_stamp
    if not os.path.isdir(tensorboard_path):
        os.mkdir(tensorboard_path)
        #按时间把不同文件的tensorboard扔到不同的位置下
    writer=SummaryWriter(tensorboard_path)
    #print the arguments
    print("let's see the argument")
    # print(args.__dict__)
    print("masktype:",masktype)
    print("GPU NUMS:",GPU_NUMS)
    print("BATCH_SIZE:",BATCH_SIZE)
    print("BATCH_SIZE per GPU:",BATCH_SIZE/GPU_NUMS)
    #load data
    print('loading data...')
    train_dataset=CasualUNet_Dataset(data_type='train')
    valid_dataset=CasualUNet_Dataset(data_type='valid')
    # test_dataset=myDataset(data_type='test')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    #get model
    model_Casual = CasualUNet()
    #多卡部分，加入一个nn.DataParallel
    model_Casual=nn.DataParallel(model_Casual,device_ids=device_ids)
    
    # loss_func=my_Loss()
    # valid_loss_func=my_Loss()
    loss_func=SingleSrcNegSDR("sisdr",reduction='mean')
    valid_loss_func=SingleSrcNegSDR("sisdr",reduction='mean')
    if torch.cuda.is_available():
        model_Casual=model_Casual.to(device)
    print("# model_CasualUNet parameters:",sum(param.numel() for param in model_Casual.parameters()))
    
    model_Casual_optimizer=optim.Adam(model_Casual.parameters(),lr=LR,betas=ADAM_BETA)
    #多卡在optimizer也添加
    # model_Casual_optimizer=nn.DataParallel(model_Casual_optimizer,device_ids=device_ids)
    if not os.path.isdir("./models"):
            os.mkdir("./models")
    if not os.path.isdir("./models/checkpoint"):
            os.mkdir("./models/checkpoint")
    checkpoint_path="./models/checkpoint/current_checkpoint.pth"
    model_path='./models/'
    if not os.path.exists(checkpoint_path):
        IF_RESUME_BREAK=False
    start_epoch=0
    best_loss=1e6
    best_epoch=-1
    nums=0
    if IF_RESUME_BREAK:
        checkpoint=torch.load(checkpoint_path)
        model_Casual.load_state_dict(checkpoint['net'])
        model_Casual_optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch=checkpoint['epoch']+1
        #give scheduler information
    # scheduler=lr_scheduler.StepLR(model_TAC_optimizer,15,0.5)
    # scheduler=lr_scheduler.StepLR(model_TAC_optimizer,50,0.2)
    scheduler=lr_scheduler.ReduceLROnPlateau(model_Casual_optimizer, mode="min",factor=0.5, patience=5, verbose=True,  cooldown=0, min_lr=0, eps=1e-08)
    for epoch in range(start_epoch,NUM_EPOCHS):
        train_loss=0
        train_bar=tqdm(train_data_loader)
        model_Casual.train()
        start_time=time.time()
        nums=0
        for mixture, target_spk_batch, target_noise_batch,ref_batch in train_bar:
            nums=nums+1
            if torch.cuda.is_available():
                mixture=mixture.to(device)
                target_spk_batch=target_spk_batch.to(device)
                target_noise_batch=target_noise_batch.to(device)
                ref_batch=ref_batch.to(device)
            
            output_batch=model_Casual(mixture,masktype)
            # output_spk1=output_batch[:,0:1,:]
            # output_spk2=output_batch[:,1:2,:]
            target_speech=target_spk_batch[:,0,:]
            target_noise=target_noise_batch[:,0,:]
            # my_maeloss=loss_func(target_speech,target_noise,output_batch)
            my_maeloss=loss_func(output_batch,target_speech)
            train_loss+=my_maeloss.detach().item()
            model_Casual.zero_grad()
            my_maeloss.backward()
            model_Casual_optimizer.step()
            #多卡的step过程需要稍微注意
            # model_Casual_optimizer.module.step()
            #scheduler step
            
            train_bar.set_description(
                'Training Epoch {}:SI-SDR LOSS {:.12f}:Train_Loss_sum{:.12f}'
                    .format(epoch,my_maeloss.item(),train_loss,))
                    #scheduler.get_last_lr() is not available in reduce plateau
        print("training finished this epoch")
        
        train_loss=train_loss/nums
        print("[Trainning] [%d/%d], Elipse Time: %4f, Train Loss: %4f" % (
                epoch , NUM_EPOCHS, time.time() - start_time, train_loss))
        checkpoint = {
                    "net": model_Casual.state_dict(),
                    'optimizer': model_Casual_optimizer.state_dict(),
                    "epoch": epoch
                }
        torch.save(checkpoint, checkpoint_path)
        # del input_real,input_imag,output_real,output_imag
        gc.collect()
        #time for cross validation
        valid_loss=0
        valid_bar=tqdm(valid_data_loader)
        model_Casual.eval()
        start_time=time.time()
        nums=0
        with torch.no_grad():
            for mixture, target_spk_batch, target_noise_batch,ref_batch in valid_bar:
                nums=nums+1
                # print("val_input_real:",val_input_real)
                # time.sleep(5)
                # print("val_input_imag:",val_input_imag)
                # time.sleep(5)
                # print("val_target_real:",val_target_real)
                # time.sleep(5)
                # print("val_target_imag:",val_target_imag)
                # time.sleep(5)
                # print("val_target_ene:",torch.sum(val_target_real**2))
                # time.sleep(10)
                if torch.cuda.is_available():
                    mixture=mixture.to(device)
                    target_spk_batch=target_spk_batch.to(device)
                    target_noise_batch=target_noise_batch.to(device)
                    ref_batch=ref_batch.to(device)
                # model_TAC.zero_grad()
                output_batch=model_Casual(mixture,masktype)
                target_speech=target_spk_batch[:,0,:]
                target_noise=target_noise_batch[:,0,:]
                # valid_maeloss=valid_loss_func(target_speech,target_noise,output_batch)
                valid_maeloss=valid_loss_func(output_batch,target_speech)
                valid_loss+=valid_maeloss.detach().item()
                #my_mseloss.backward()
                #model_TAC_optimizer.step()
                valid_bar.set_description(
                    'Validation Epoch {}:SI-SDR LOSS {:.12f}:Valid_Loss_sum{:.12f}'
                        .format(epoch,valid_maeloss.item(),valid_loss))
            print("validation finished this epoch")
            valid_loss=valid_loss/nums
            scheduler.step(valid_loss)
            print("[validation] [%d/%d], Elipse Time: %4f, valid Loss: %4f" % (
                epoch , NUM_EPOCHS, time.time() - start_time, valid_loss))
        if(valid_loss<=best_loss):
            best_loss=valid_loss
            best_epoch=epoch
            best_unet_path = os.path.join(model_path, 'best-valid-multi.pkl' )
            # 多卡的保存
            # best_unet = model_Casual.state_dict()
            best_unet = model_Casual.module.state_dict()
            # 多卡的保存
            
            print('Find better valid model, Best  model loss : %.4f\n' % ( valid_loss))
            torch.save(best_unet, best_unet_path)
            # 多卡的保存
            
        gc.collect()
        if(epoch%5==0):
            five_unet_path = os.path.join(model_path, 'epoch_%d.pkl' % (
                     epoch))
            # five_unet = model_Casual.state_dict()
            five_unet = model_Casual.module.state_dict()
            torch.save(five_unet, five_unet_path)
        print("the best epoch is:",best_epoch)
        #tensorboard add scalar(loss)
        writer.add_scalar("train loss",train_loss,epoch)
        writer.add_scalar("valid loss",valid_loss,epoch)
        gc.collect()
        #finish tensorboard add scalar
    print("the training process all done")  

