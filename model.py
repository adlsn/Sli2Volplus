import torch
import torch.nn as nn
import torch.nn.functional as F

from CorrespondingNet import ResNet18_original_size, one_hot


class OEG_CFN(nn.Module):
    def __init__(self, in_channels=1, is_training=True, R=6, refine_mode=False):
        super(OEG_CFN, self).__init__()
        """
        in_channels: number of channels of the input 2D slice, e.g. gray=1, RGB=3, etc
        is_training: True if training mode
        R: radius of window for local attention
        """
        self.is_training = is_training
        self.R = R
        self.refine_mode = refine_mode

        self.feature_extraction = ResNet18_original_size(in_channels)
        self.post_convolution = nn.Conv2d(128, 64, 3, 1, 1)
        self.feature_extraction_2 = ResNet18_original_size(in_channels)
        self.post_convolution_2 = nn.Conv2d(128, 64, 3, 1, 1)

        # Use smaller R for faster training
        self.corr_recontruction = Corr_recontruction(R=self.R, is_training=self.is_training,
                                                     refine_mode=self.refine_mode)

    def forward(self, slice1_input, slice2_input, slice1, pseudo_label1, pseudo_label2):
        """
        Compute the features of slice1 and slice2, use their affinity matrix to reconstruct slice2 from slice1
        """
        _, _, _, _, feats_r = self.feature_extraction(slice1_input)
        _, _, _, _, feats_t = self.feature_extraction(slice2_input)

        feats_r = self.post_convolution(feats_r)
        feats_t = self.post_convolution(feats_t)

        _, _, _, _, pl1_feature = self.feature_extraction_2(pseudo_label1)
        _, _, _, _, pl2_feature = self.feature_extraction_2(pseudo_label2)

        pl1_feature = self.post_convolution_2(pl1_feature)
        pl2_feature = self.post_convolution_2(pl2_feature)

        slice2_reconstructed = self.corr_recontruction(feats_r, feats_t, slice1, pl1_feature, pl2_feature)

        return slice2_reconstructed


class Corr_recontruction(nn.Module):
    def __init__(self, R=6, is_training=True, refine_mode=False):
        super(Corr_recontruction, self).__init__()
        self.R = R  # half window size
        self.is_training = is_training
        self.refine_mode = refine_mode

        self.correlation_sampler_custom = Correlation_layer(
            padding=self.R, train_mode=self.is_training, refine_mode=self.refine_mode
        )

    def prep(self, image):
        _, c, _, _ = image.size()

        if self.is_training:
            x = image.float()
        else:
            x = image.float()
            x = one_hot(x.long(),
                        7).float()  # to one-hot label, 7 is just random, as long as it is larger than the total number of classes

        return x

    def forward(self, feats_r, feats_t, img_r, plf_1, plf_2):
        """
        Warp img_r to img_t. Using similarity computed with feats_r and feats_t
        """

        b, _, h, w = feats_t.size()
        img_r = self.prep(img_r)
        _, c, _, _ = img_r.size()

        corrs = [torch.zeros(1)]
        corrs_img = [torch.zeros(1)]
        for ind in range(1):
            corrs[ind], corrs_img[ind] = self.correlation_sampler_custom(feats_t, feats_r, img_r, plf_1, plf_2)

        corr = torch.cat(corrs, 1)
        corr = F.softmax(corr, dim=1)
        corrs_img = torch.cat(corrs_img, 1)

        img_t = []
        for start in range(c):
            img_t.append((corr * corrs_img[:, start::c, :, :]).sum(1, keepdim=True))
        img_t = torch.cat(img_t, 1)

        return img_t


class Correlation_layer(nn.Module):
    def __init__(self, padding=20, train_mode=True, refine_mode=False):
        super(Correlation_layer, self).__init__()
        """
        Inspired and modified from https://github.com/limacv/CorrelationLayer/blob/master/correlation_torch.py
        This module considers the pseudo label and the feature map of the input image
        """
        self.pad = padding
        self.max_displacement = padding
        self.train_mode = train_mode
        self.refine_mode = refine_mode

    def forward(self, x_1, x_2, img, p_1, p_2):
        """
        Arguments
        ---------
        x_1 : 4D torch.Tensor (bathch channel height width)
        x_2 : 4D torch.Tensor (bathch channel height width)
        p_1 : 4D torch.Tensor (bathch channel height width) pseudo label feature
        p_2 : 4D torch.Tensor (bathch channel height width) pseudo label feature
        """

        x_1 = x_1.transpose(1, 2).transpose(2, 3)  # b h w c
        x_2 = F.pad(x_2, tuple([self.pad for _ in range(4)])).transpose(1, 2).transpose(2, 3)
        img = F.pad(img, tuple([self.pad for _ in range(4)]))
        p_1 = p_1.transpose(1, 2).transpose(2, 3)
        p_2 = F.pad(p_2, tuple([self.pad for _ in range(4)])).transpose(1, 2).transpose(2, 3)

        if self.train_mode:
            x_1 = torch.cat((x_1, p_1), 3)
            x_2 = torch.cat((x_2, p_2), 3)
        if self.refine_mode:
            x_1 = torch.cat((x_1, p_1), 3)
            x_2 = torch.cat((x_2, p_2), 3)

        out_vb = torch.zeros(1)
        out_img = torch.zeros(1)
        _y = 0
        _x = 0
        for _y in range(self.max_displacement * 2 + 1):
            for _x in range(self.max_displacement * 2 + 1):
                c_out = (torch.sum(x_1 * x_2[:, _y:_y + x_1.size(1),
                                         _x:_x + x_1.size(2), :], 3, keepdim=True)).transpose(2, 3).transpose(1,
                                                                                                              2)  # b c h w
                # c_out = self.post_cnn((x_1*x_2[:, _y:_y+x_1.size(1), _x:_x+x_1.size(2),:]).transpose(2,3).transpose(1,2)) #b c h w
                out_img = torch.cat((out_img, img[:, :, _y:_y + x_1.size(1), _x:_x + x_1.size(2)]), 1) if len(
                    out_img.size()) != 1 else img[:, :, _y:_y + x_1.size(1), _x:_x + x_1.size(2)]
                out_vb = torch.cat((out_vb, c_out), 1) if len(out_vb.size()) != 1 else c_out

        return out_vb, out_img
