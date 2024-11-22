"""
This file is a modified version based on Sli2Vol.
@author: adlsn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, kernel_size=3, activation=F.relu):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.activation = activation

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ResNet18_original_size(nn.Module):
    def __init__(self, in_ch=1):
        super(ResNet18_original_size, self).__init__()
        self.inchannel = 16
        start_channel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, start_channel, kernel_size=7, stride=1, padding=3, bias=False),    # stride=1 because we don't want downsampling
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(ResidualBlock, start_channel, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, start_channel*2, 2, stride=1)
        self.layer3 = self.make_layer(ResidualBlock, start_channel*4, 2, stride=1)
        self.layer4 = self.make_layer(ResidualBlock, start_channel*8, 2, stride=1)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

def one_hot(labels, C):
    one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3))
    if labels.is_cuda: one_hot = one_hot.cuda()

    target = one_hot.scatter_(1, labels, 1)
    if labels.is_cuda: target = target.cuda()

    return target

class Correspondence_Flow_Net(nn.Module):
    def __init__(self, in_channels=1, is_training=True, R=6, refine_mode=False):
        super(Correspondence_Flow_Net, self).__init__()
        """
        in_channels: number of channels of the input 2D slice, e.g. gray=1, RGB=3, etc
        is_training: True if training mode
        R: radius of window for local attention
        """
        self.is_training=is_training
        self.R = R
        self.refine_mode = refine_mode
        
        self.feature_extraction = ResNet18_original_size(in_channels)
        self.post_convolution = nn.Conv2d(128, 64, 3, 1, 1)

        # Use smaller R for faster training
        self.corr_recontruction = Corr_recontruction(R=self.R, is_training=self.is_training, refine_mode=self.refine_mode)
            
    

    def forward(self, slice1_input, slice2_input, slice1, pseudo_label1, pseudo_label2):
        """
        Compute the features of slice1 and slice2, use their affinity matrix to reconstruct slice2 from slice1
        """
        _,_,_,_,feats_r = self.feature_extraction(slice1_input)
        _,_,_,_,feats_t = self.feature_extraction(slice2_input)
        
        feats_r = self.post_convolution(feats_r)
        feats_t = self.post_convolution(feats_t)
        
        _,_,_,_,pl1_feature = self.feature_extraction(pseudo_label1)
        _,_,_,_,pl2_feature = self.feature_extraction(pseudo_label2)
        
        pl1_feature = self.post_convolution(pl1_feature)
        pl2_feature = self.post_convolution(pl2_feature)
        
        slice2_reconstructed = self.corr_recontruction(feats_r, feats_t, slice1, pl1_feature, pl2_feature)
        
        return slice2_reconstructed
        
        
class Corr_recontruction(nn.Module):
    def __init__(self, R=6, is_training=True, refine_mode=False):
        super(Corr_recontruction, self).__init__()
        self.R = R  # half window size
        self.is_training = is_training
        self.refine_mode = refine_mode

        self.correlation_sampler_custom = CorrelationLayer(
            padding=self.R, train_mode=self.is_training, refine_mode=self.refine_mode
            )

    def prep(self, image):
        _,c,_,_ = image.size()
        
        if self.is_training:
            x = image.float()
        else:
            x = image.float()
            x = one_hot(x.long(), 7).float() # to one-hot label, 7 is just random, as long as it is larger than the total number of classes

        return x

    def forward(self, feats_r, feats_t, img_r, plf_1, plf_2):
        """
        Warp img_r to img_t. Using similarity computed with feats_r and feats_t
        """
        
        b,_,h,w = feats_t.size()
        img_r = self.prep(img_r)
        _,c,_,_ = img_r.size()
        
        corrs = [torch.zeros(1)]
        corrs_img = [torch.zeros(1)]
        for ind in range(1):
            corrs[ind], corrs_img[ind] = self.correlation_sampler_custom(feats_t, feats_r, img_r, plf_1, plf_2)
        
        corr = torch.cat(corrs, 1)  
        corr = F.softmax(corr, dim=1)
        corrs_img = torch.cat(corrs_img, 1)
        
        img_t = []
        for start in range(c):
            img_t.append((corr*corrs_img[:,start::c,:,:]).sum(1,keepdim=True))
        img_t = torch.cat(img_t, 1)
        
        return img_t
        

class CorrelationLayer(nn.Module):
    def __init__(self, padding=20, train_mode=True, refine_mode=False):
        super(CorrelationLayer,self).__init__()
        """
        Inspired and modified from https://github.com/limacv/CorrelationLayer/blob/master/correlation_torch.py
        Do not need any cuda implementation like https://github.com/NVIDIA/flownet2-pytorch
        But the efficiency is slightly compromised
        ***Modification: add features extracted by pseudo label into the key, query matrix***
        """
        self.pad = padding
        self.max_displacement = padding
        self.train_mode = train_mode
        self.refine_mode = refine_mode
        # self.post_cnn = nn.Conv2d(128, 1, 1, 1, 0) # TODO: another version
        # self.post_cnn2 = nn.Conv2d(128, 1, 1, 1, 0) # TODO: another version
        


    def forward(self, x_1, x_2, img, p_1, p_2):
        """
        Arguments
        ---------
        x_1 : 4D torch.Tensor (bathch channel height width)
        x_2 : 4D torch.Tensor (bathch channel height width)
        p_1 : 4D torch.Tensor (bathch channel height width) pseudo label feature
        p_2 : 4D torch.Tensor (bathch channel height width) pseudo label feature
        """
        
        x_1 = x_1.transpose(1,2).transpose(2,3) #b h w c
        x_2 = F.pad(x_2, tuple([self.pad for _ in range(4)])).transpose(1,2).transpose(2,3)
        img = F.pad(img, tuple([self.pad for _ in range(4)]))
        p_1 = p_1.transpose(1,2).transpose(2,3)
        p_2 = F.pad(p_2, tuple([self.pad for _ in range(4)])).transpose(1,2).transpose(2,3)
        
        if self.train_mode:
            x_1 = torch.cat((x_1, p_1), 3)
            x_2 = torch.cat((x_2, p_2), 3)
        if self.refine_mode:
            pass
        
        # TODO can be optimize
        out_vb = torch.zeros(1)
        out_img = torch.zeros(1)
        _y=0
        _x=0
        for _y in range(self.max_displacement*2+1):
            for _x in range(self.max_displacement*2+1):
                c_out = (torch.sum(x_1*x_2[:, _y:_y+x_1.size(1),
                                          _x:_x+x_1.size(2),:],3, keepdim=True)).transpose(2,3).transpose(1,2) #b c h w
                # c_out = self.post_cnn((x_1*x_2[:, _y:_y+x_1.size(1), _x:_x+x_1.size(2),:]).transpose(2,3).transpose(1,2)) #b c h w
                out_img = torch.cat((out_img,img[:, :, _y:_y+x_1.size(1), _x:_x+x_1.size(2)]),1) if len(out_img.size())!=1 else img[:, :, _y:_y+x_1.size(1), _x:_x+x_1.size(2)]
                out_vb = torch.cat((out_vb,c_out),1) if len(out_vb.size())!=1 else c_out
                
        return out_vb, out_img
    
    # def forward(self, x_1, x_2, img, p_1, p_2):
    #     """
    #     Arguments
    #     ---------
    #     x_1 : 4D torch.Tensor (bathch channel height width)
    #     x_2 : 4D torch.Tensor (bathch channel height width)
    #     p_1 : 4D torch.Tensor (bathch channel height width) pseudo label feature
    #     p_2 : 4D torch.Tensor (bathch channel height width) pseudo label feature
    #     """
    #     x_11 = torch.empty_like(x_1).copy_(x_1)
    #     x_22 = torch.empty_like(x_2).copy_(x_2)
    #     x_22 = x_22.transpose(1,2).transpose(2,3)
    #     x_11 = F.pad(x_11, tuple([self.pad for _ in range(4)])).transpose(1,2).transpose(2,3) 
    #     p_11 = torch.empty_like(p_1).copy_(p_1)
    #     p_22 = torch.empty_like(p_2).copy_(p_2)
    #     p_11 = F.pad(p_11, tuple([self.pad for _ in range(4)])).transpose(1,2).transpose(2,3)
    #     p_22 = p_22.transpose(1,2).transpose(2,3) # TODO: before is bi-directional method.
        
    #     x_1 = x_1.transpose(1,2).transpose(2,3) #b h w c
    #     x_2 = F.pad(x_2, tuple([self.pad for _ in range(4)])).transpose(1,2).transpose(2,3)
    #     img = F.pad(img, tuple([self.pad for _ in range(4)]))
    #     p_1 = p_1.transpose(1,2).transpose(2,3)
    #     p_2 = F.pad(p_2, tuple([self.pad for _ in range(4)])).transpose(1,2).transpose(2,3)
        
    #     if self.train_mode:
    #         x_1 = torch.cat((x_1, p_1), 3)
    #         x_2 = torch.cat((x_2, p_2), 3)
    #         x_11 = torch.cat((x_11, p_11), 3)
    #         x_22 = torch.cat((x_22, p_22), 3)
    #     # else:
    #     #     p_1 = torch.zeros_like(p_1).cuda()
    #     #     p_2 = torch.zeros_like(p_2).cuda()
    #     #     p_11 = torch.zeros_like(p_11).cuda()
    #     #     p_22 = torch.zeros_like(p_22).cuda()
    #     #     x_1 = torch.cat((x_1, p_1), 3)
    #     #     x_2 = torch.cat((x_2, p_2), 3)
    #     #     x_11 = torch.cat((x_11, p_11), 3)
    #     #     x_22 = torch.cat((x_22, p_22), 3)
        
    #     # TODO can be optimize
    #     out_vb = torch.zeros(1)
    #     out_img = torch.zeros(1)
    #     bi_out_vb = torch.zeros(1)
    #     _y=0
    #     _x=0
    #     for _y in range(self.max_displacement*2+1):
    #         for _x in range(self.max_displacement*2+1):
    #             c_out = (torch.sum(x_1*x_2[:, _y:_y+x_1.size(1),
    #                                       _x:_x+x_1.size(2),:],3, keepdim=True)).transpose(2,3).transpose(1,2) #b c h w
    #             # c_out = self.post_cnn((x_1*x_2[:, _y:_y+x_1.size(1), _x:_x+x_1.size(2),:]).transpose(2,3).transpose(1,2)) #b c h w
    #             out_img = torch.cat((out_img,img[:, :, _y:_y+x_1.size(1), _x:_x+x_1.size(2)]),1) if len(out_img.size())!=1 else img[:, :, _y:_y+x_1.size(1), _x:_x+x_1.size(2)]
    #             out_vb = torch.cat((out_vb,c_out),1) if len(out_vb.size())!=1 else c_out
        
    #     # bi-directional forward     
           
    #     for _y in range(self.max_displacement*2+1):
    #         for _x in range(self.max_displacement*2+1):
    #             # bi_c_out = self.post_cnn2((x_22*x_11[:, _y:_y+x_22.size(1),
    #             #                           _x:_x+x_22.size(2),:]).transpose(2,3).transpose(1,2)) #b c h w
    #             bi_c_out = (torch.sum(x_22*x_11[:, _y:_y+x_22.size(1),
    #                                       _x:_x+x_22.size(2),:],3,keepdim=True)).transpose(2,3).transpose(1,2) #b c h w
    #             bi_out_vb = torch.cat((bi_out_vb,bi_c_out),1) if len(bi_out_vb.size())!=1 else bi_c_out
                
    #     # out_vb *= bi_out_vb
        
    #     out_vb = torch.cat((out_vb,bi_out_vb),1)
    #     out_img = torch.cat((out_img,out_img),1)
                

    #     return out_vb, out_img