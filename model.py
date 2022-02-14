""" Yangang Cao 2021.12.1 15:30"""
import torch
import torch.nn as nn

#2022.2.10修改：我们先把学习的mask分为幅度和相位两部分，对幅度部分再过一个tanh试试
K=8
epsilon=1e-6
class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            # nn.Dropout2d(0.5),
            # nn.LeakyReLU(0.3,inplace=True)
            nn.PReLU()
        )

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        return self.conv(x)


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            # nn.Dropout2d(0.5),
            # nn.LeakyReLU(0.3,inplace=True)
            nn.PReLU()
        )

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        return self.conv(x)


class CasualUNet(nn.Module):
    """
    Input: [B, C, F, T]
    Output: [B, C, T, F]
    """
    def __init__(self):
        super(CasualUNet, self).__init__()
        self.conv_block_1 = CausalConvBlock(8, 32, (8, 2), (2, 1))
        self.conv_block_2 = CausalConvBlock(32, 32, (6, 2), (2, 1))
        self.conv_block_3 = CausalConvBlock(32, 64, (7, 2), (2, 1))
        self.conv_block_4 = CausalConvBlock(64, 64, (6, 2), (2, 1))
        self.conv_block_5 = CausalConvBlock(64, 96, (6, 2), (2, 1))
        self.conv_block_6 = CausalConvBlock(96, 96, (6, 2), (2, 1))
        self.conv_block_7 = CausalConvBlock(96, 128, (2, 2), (2, 1))
        self.conv_block_8 = CausalConvBlock(128, 256, (2, 2), (1, 1))

        self.tran_conv_block_1 = CausalTransConvBlock(256, 256, (2, 2), (1, 1))
        self.tran_conv_block_2 = CausalTransConvBlock(256 + 128, 128, (2, 2), (2, 1))
        self.tran_conv_block_3 = CausalTransConvBlock(128 + 96, 96, (6, 2), (2, 1))
        self.tran_conv_block_4 = CausalTransConvBlock(96 + 96, 96, (6, 2), (2, 1))
        self.tran_conv_block_5 = CausalTransConvBlock(96 + 64, 64, (6, 2), (2, 1))
        self.tran_conv_block_6 = CausalTransConvBlock(64 + 64, 64, (7, 2), (2, 1))
        self.tran_conv_block_7 = CausalTransConvBlock(64 + 32, 32,  (6, 2), (2, 1))
        self.tran_conv_block_8 = CausalTransConvBlock(32 + 32, 8,  (8, 2), (2, 1))
        self.dense = nn.Linear(514, 514)
        # self.activate_mask=nn.Tanh()

    def forward(self, x,masktype):
        self.masktype=masktype
        batch_size=x.size(0)
        num_mics=x.size(1)
        x_reshaped=x.view(batch_size*num_mics,-1)
        x_stft=torch.stft(x_reshaped,512,256,return_complex=False)
        frequency_bins=x_stft.size(1)
        time_frames=x_stft.size(2)
        x_stft_reshaped=x_stft.view(batch_size,num_mics,frequency_bins,time_frames,-1)
        #note: in pytorch 1.7,the default setting(return complex=False),torch.stft will return a matrix 
        x_stft_real=x_stft_reshaped[:,:,:,:,0]
        x_stft_imag=x_stft_reshaped[:,:,:,:,1]
        x_stft_cat=torch.cat((x_stft_real,x_stft_imag),2) #(batchsize,nummics,2*freq_bins,frame_size)
        # print("x_stft_cat shape:",x_stft_cat.shape)
        prefix_frames = torch.zeros(batch_size, num_mics, 514, 8) # K zeros prefix frames,K=8
        if x_stft_cat.is_cuda:
            prefix_frames=prefix_frames.cuda()
        x_stft_cat_new = torch.cat((prefix_frames, x_stft_cat), 3)
        # print("x_stft_cat_new shape:",x_stft_cat_new.shape)
        # e1 = self.conv_block_1(x)
        e1 = self.conv_block_1(x_stft_cat_new)
        # print("e1 shape:",e1.shape)
        e2 = self.conv_block_2(e1)
        # print("e2 shape:",e2.shape)
        e3 = self.conv_block_3(e2)
        # print("e3 shape:",e3.shape)
        e4 = self.conv_block_4(e3)
        # print("e4 shape:",e4.shape)
        e5 = self.conv_block_5(e4)
        # print("e5 shape:",e5.shape)
        e6 = self.conv_block_6(e5)
        # print("e6 shape:",e6.shape)
        e7 = self.conv_block_7(e6)
        # print("e7 shape:",e7.shape)
        e8 = self.conv_block_8(e7)
        # print("e8 shape:",e8.shape)

        d = self.tran_conv_block_1(e8)
        # print("d shape:",d.shape)
        d = self.tran_conv_block_2(torch.cat((d, e7), 1))
        # print("d shape:",d.shape)
        d = self.tran_conv_block_3(torch.cat((d, e6), 1))
        # print("d shape:",d.shape)
        d = self.tran_conv_block_4(torch.cat((d, e5), 1))
        # print("d shape:",d.shape)
        d = self.tran_conv_block_5(torch.cat((d, e4), 1))
        # print("d shape:",d.shape)
        d = self.tran_conv_block_6(torch.cat((d, e3), 1))
        # print("d shape:",d.shape)
        d = self.tran_conv_block_7(torch.cat((d, e2), 1))
        # print("d shape:",d.shape)
        d = self.tran_conv_block_8(torch.cat((d, e1), 1))
        # print("d shape:",d.shape)
        d = d.permute(0,1,3,2)
        d = self.dense(d)
        d=d[:,:,:-8,:] # strip the last K frames
        #变回去
        mask_out_new=d.permute(0,1,3,2) #(batchsize,num_mics,2*freq_bins,frame_size)
        mask_out_real=mask_out_new[:,:,:frequency_bins,:]
        mask_out_imag=mask_out_new[:,:,frequency_bins:2*frequency_bins,:]  #(batchsize,num_mics,freq_bins,frame_size)
        # print("mask out imag dim:",mask_out_imag.shape)
        #实际上，原始的操作存在问题，本身复数乘除，相当于被认为去掉了一部分，最后结果非常混乱
        #提取出原始信号（x）以及学习到的mask（mask）的幅度（mag）以及相位(phase)
        #先尝试maskR
        if(self.masktype=='R'):
            out_real=mask_out_real*x_stft_real
            out_imag=mask_out_imag*x_stft_imag
            out_stft=torch.complex(out_real,out_imag)
        elif(self.masktype=='C'):
            out_real=mask_out_real*x_stft_real-mask_out_imag*x_stft_imag
            out_imag=mask_out_real*x_stft_imag+mask_out_imag*x_stft_real
            out_stft=torch.complex(out_real,out_imag)
        elif(self.masktype=='E'):
            mask_out_mags = (mask_out_real ** 2 + mask_out_imag ** 2) ** 0.5
            real_phase=mask_out_real/(1e-8+mask_out_mags)
            imag_phase=mask_out_imag/(1e-8+mask_out_mags)
            mask_out_phase=torch.atan2(imag_phase,real_phase)
            mask_out_mags=torch.tanh(mask_out_mags)
            x_stft_mags=torch.sqrt(x_stft_real ** 2 + x_stft_imag ** 2 + 1e-8)
            x_real_phase=x_stft_real/(1e-8+x_stft_mags)
            x_imag_phase=x_stft_imag/(1e-8+x_stft_mags)
            x_stft_phase=torch.atan2(x_imag_phase,x_real_phase)
            est_mags=mask_out_mags*x_stft_mags
            est_phase=mask_out_phase+x_stft_phase
            out_real=est_mags*torch.cos(est_phase)
            out_imag=est_mags*torch.sin(est_phase)
            out_stft=torch.complex(out_real,out_imag)
            ##尽量先分别算幅度和相位，再分别算实部虚部，避免让网络计算torch.exp(1j*xx)的反向传播
       
        # x_stft_real+=epsilon
        # x_stft_imag+=epsilon
        # mask_out_real+=epsilon
        # mask_out_imag+=epsilon
        # ###先给他们都加一个epsilon，避免出现0,0作为被除数会出现nan
        # x_mag=x_stft_real**2+x_stft_imag**2
        # x_mag=torch.sqrt(x_mag)
        # mask_out_mag=mask_out_real**2+mask_out_imag**2
        # mask_out_mag=torch.sqrt(mask_out_mag)
        # x_phase=torch.arctan(x_stft_imag/x_stft_real)
        # mask_out_phase=torch.arctan(mask_out_imag/mask_out_real)
      
        #幅度mask过tanh
        # mask_out_mag=self.activate_mask(mask_out_mag)
        # print("mask out mag dim:",mask_out_mag.shape)
        #mask应用
        # out_stft=(x_stft_real+1j*x_stft_imag)*(mask_out_real+1j*mask_out_imag)
        # out_stft=mask_out_mag*x_mag*torch.exp((x_phase+mask_out_phase)*1j)
        # print("out stft1:",out_stft1)
        
        out_stft=torch.sum(out_stft,1)
        
       
     
        out_wave=torch.istft(out_stft,512,256) #(batchsize,samplerate*time_length=4*16000)
       
        return out_wave


if __name__ == '__main__':
    model=CasualUNet()
    a=torch.randn(64,8,64000)
    out=model(a)
    print("a size:",a.shape)
    print("out size:",out.shape)