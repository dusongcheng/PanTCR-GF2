import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.nn.functional import cosine_similarity
from sewar.full_ref import uqi
from scipy.ndimage import uniform_filter
import os
import cv2
from scipy import ndimage


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad
    
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

def save_logfile(save_root, epoch, train_loss, valid_loss, test_loss, psnr, ssim):
    log_path = save_root
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = os.path.join(log_path, 'records.txt')
    open_type = 'a' if os.path.exists(log_path)else 'w'
    log_file = open(log_path, open_type)
    log = 'Epoch {:02d}: train_loss {:.4f}, valid_loss {:.4f}, test_loss {:.4f}, psnr {:.4f}, ssim {:.4f}'.format(epoch, train_loss, valid_loss, test_loss, psnr, ssim)
    log_file.write(str(log) + '\n')


def record_loss(loss_csv,epoch, train_loss, valid_loss, test_loss, psnr, ssim):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, train_loss, valid_loss, test_loss, psnr, ssim))
    loss_csv.flush()    
    loss_csv.close
    
def show(epoch, srf, srf_g, psf, psf_g):
    srf = np.array(srf.data.cpu())
    srf_g = np.array(srf_g.data.cpu())
    psf = np.array(psf.data.cpu())
    psf_g = np.array(psf_g.data.cpu())
    # show SRF
    channel = range(31)
    plt.figure(figsize=(10, 6), facecolor='lightgray', edgecolor='black')
    plt.plot(channel, srf[0,:], marker='o', linestyle='--', color='b')
    plt.plot(channel, srf_g[0,:], marker='o', linestyle='-', color='b')
    plt.plot(channel, srf[1,:], marker='o', linestyle='--', color='g')
    plt.plot(channel, srf_g[1,:], marker='o', linestyle='-', color='g')
    plt.plot(channel, srf[2,:], marker='o', linestyle='--', color='r')
    plt.plot(channel, srf_g[2,:], marker='o', linestyle='-', color='r')
    plt.title('Spectral Response Curve')
    plt.xlabel('Spectral')
    plt.ylabel('Response')
    plt.grid(True)
    plt.savefig('models/fig/src_epoch'+str(epoch)+'.png')
    #show(PSF)
    plt.figure(figsize=(10, 4), facecolor='lightgray', edgecolor='black')
    plt.subplot(131), plt.imshow(psf, cmap='hot', interpolation='nearest'), plt.title('PSF')
    plt.subplot(132), plt.imshow(psf_g, cmap='hot', interpolation='nearest'), plt.title('PSF_GT')
    plt.subplot(133), plt.imshow(np.abs(psf-psf_g), cmap='hot', interpolation='nearest'), plt.title('PSF_error')
    plt.savefig('models/fig/psf_epoch'+str(epoch)+'.png')
    plt.close('all')


def load_model(model, model_name, model_var='Model_stage1'):
    model_param = torch.load(model_name, weights_only=True)[model_var]
    model_dict = {}
    for k1, k2 in zip(model.state_dict(), model_param):
        model_dict[k1] = model_param[k2]
    model.load_state_dict(model_dict)
    return model.cuda()

def PSNR_SSIM_cal(gt, rec):
    gt = gt.detach().cpu().numpy()
    rec = rec.detach().cpu().numpy()
    psnr = cal_psnr(gt[0,:,:,:], rec[0,:,:,:])
    gt = np.transpose(gt,(0,2,3,1))[0,:,:,:]
    rec = np.transpose(rec,(0,2,3,1))[0,:,:,:]
    ssim = compare_ssim(gt, rec, data_range=1., channel_axis=-1)
    return psnr, np.mean(np.array(ssim))

def cal_cos_loss(l1, l2):
    l1 = l1.view(-1)
    l2 = l2.view(-1)
    similarity = cosine_similarity(l1, l2, dim=-1, channel_axis=-1)
    return similarity


def cal_decomp_loss(RGB_spatial_spectral, RGB_spatial, HSI_spatial_spectral, HSI_spectral):
    positive = torch.exp(cal_cos_loss(RGB_spatial_spectral, HSI_spatial_spectral))
    negative = torch.exp(cal_cos_loss(RGB_spatial_spectral, RGB_spatial)) + torch.exp(cal_cos_loss(HSI_spatial_spectral, HSI_spectral)) + torch.exp(cal_cos_loss(RGB_spatial, HSI_spectral))
    decomp_loss = -torch.log(positive/(positive+negative))
    return decomp_loss


def cal_psnr(label, output):

    img_c, img_w, img_h = label.shape
    ref = label.reshape(img_c, -1)
    tar = output.reshape(img_c, -1)
    msr = np.mean((ref - tar) ** 2, 1)
    max1 = np.max(ref, 1)

    psnrall = 10 * np.log10(1 / msr)
    out_mean = np.mean(psnrall)
    # return out_mean, max1
    return out_mean


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeter_valid(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = np.zeros([1,6])
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*np.array(n)
        self.count += n
        self.avg = self.sum / self.count

class Loss_valid(nn.Module):
    def __init__(self, scale=4):
        super(Loss_valid, self).__init__()
        self.scale=scale

    def forward(self, label_image, rec_image):
        self.label = label_image
        self.output = rec_image
        self.output = np.clip(self.output, 0, 1)
        valid_error = np.zeros([1, 6])

        valid_error[0, 0] = self.ssim()
        valid_error[0, 1] = self.cal_rmse()
        valid_error[0, 2] = self.cal_psnr()
        valid_error[0, 3] = self.cal_ergas()
        valid_error[0, 4] = self.sam()
        valid_error[0, 5] = self.cal_uqi()
        return valid_error

    def cal_mrae(self):
        error = np.abs(self.output - self.label) / self.label
        # error = torch.abs(outputs - label)
        mrae = np.mean(error.reshape(-1))
        return mrae

    def cal_rmse(self):
        rmse = np.sqrt(np.mean((self.label-self.output)**2))
        return rmse

    def cal_psnr(self):
        
        assert self.label.ndim == 3 and self.output.ndim == 3

        img_c, img_w, img_h = self.label.shape
        ref = self.label.reshape(img_c, -1)
        tar = self.output.reshape(img_c, -1)
        msr = np.mean((ref - tar) ** 2, 1)
        max1 = np.max(ref, 1)

        psnrall = 10 * np.log10(1 / msr)
        out_mean = np.mean(psnrall)
        # return out_mean, max1
        return out_mean

    def cal_ergas(self, scale=4):
        d = self.label - self.output
        ergasroot = 0
        for i in range(d.shape[0]):
            ergasroot = ergasroot + np.mean(d[i, :, :] ** 2) / np.mean(self.label[i, :, :]) ** 2
        ergas = (100 / scale) * np.sqrt(ergasroot/(d.shape[0]+1))
        return ergas

    def cal_sam(self):
        assert self.label.ndim == 3 and self.label.shape == self.label.shape

        c, w, h = self.label.shape
        x_true = self.label.reshape(c, -1)
        x_pred = self.output.reshape(c, -1)

        x_pred[np.where((np.linalg.norm(x_pred, 2, 1)) == 0),] += 0.0001

        sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))

        sam = np.arccos(sam) * 180 / np.pi
        # sam = np.arccos(sam)
        mSAM = sam.mean()
        var_sam = np.var(sam)
        # return mSAM, var_sam
        return mSAM

    def cal_ssim(self, data_range=1, multidimension=False):
        """
        :param x_true:
        :param x_pred:
        :param data_range:
        :param multidimension:
        :return:
        """
        mssim = [
            compare_ssim(X=self.label[i, :, :], Y=self.output[i, :, :], data_range=data_range, multidimension=multidimension)
            for i in range(self.label.shape[0])]
        return np.mean(mssim)

    def cal_uqi(self):
        fout = np.transpose(self.output, [1,2,0])
        hsi_g = np.transpose(self.label, [1,2,0])
        uqi_ = uqi(hsi_g, fout)
        return uqi_

    def ssim(self):
        fout_0 = np.transpose(self.output, [1,2,0])
        hsi_g_0 = np.transpose(self.label, [1,2,0])
        ssim_result = compare_ssim(fout_0, hsi_g_0, data_range=1., channel_axis=-1)
        return ssim_result
    
    def psnr(self):
        fout = self.output
        hsi_g = self.label
        psnr_g = []
        for i in range(31):
            psnr_g.append(compare_psnr(hsi_g[i,:,:],fout[i,:,:]))
        return np.mean(np.array(psnr_g))

    def sam(self):
        """
        cal SAM between two images
        :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
        :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
        :return: Spectral Angle Mapper between `recovered` and `groundTruth`.
        """
        groundTruth = np.transpose(self.label, [1,2,0])
        recovered = np.transpose(self.output, [1,2,0])
        assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

        nom = np.sum(groundTruth * recovered, 2)
        denom1 = np.sqrt(np.sum(groundTruth**2, 2))
        denom2 = np.sqrt(np.sum(recovered ** 2, 2))
        sam = np.arccos(np.divide(nom, denom1*denom2 + np.finfo(np.float64).eps).clip(min=0, max=1))
        sam = np.divide(sam, np.pi) * 180.0
        sam = np.mean(sam)

        return sam
    


def gaussian2d(N, std):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std) ** 2) * np.exp(-0.5 * (t2 / std) ** 2)
    return w

def kaiser2d(N, beta):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0
    return w

def fir_filter_wind(Hd, w):
    """
    compute fir (finite impulse response) filter with window method
    Hd: desired freqeuncy response (2D)
    w: window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = h / np.sum(h)
    return h

def GNyq2win(GNyq, scale=4, N=41):
    """Generate a 2D convolutional window from a given GNyq
    GNyq: Nyquist frequency
    scale: spatial size of PAN / spatial size of MS
    """
    # fir filter with window method
    fcut = 1 / scale
    alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = fir_filter_wind(Hd, w)
    return np.real(h)

def _qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    assert block_size > 1, 'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size ** 2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size / 2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1_ ** 2, -1, window)[pad_topleft:-pad_bottomright,
                pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_ ** 2, -1, window)[pad_topleft:-pad_bottomright,
                pad_topleft:-pad_bottomright] - mu2_sq
    #    print(mu1_mu2.shape)
    # print(sigma2_sq.shape)
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright,
              pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0

    #    print(np.min(sigma1_sq + sigma2_sq), np.min(mu1_sq + mu2_sq))

    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx] = ((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
            (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])

    #    print(np.mean(qindex_map))

    #    idx = ((sigma1_sq + sigma2_sq) == 0) * ((mu1_sq + mu2_sq) != 0)
    #    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    #    # sigma !=0 and mu == 0
    #    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) == 0)
    #    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    #    # sigma != 0 and mu != 0
    #    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) != 0)
    #    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
    #        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])

    return np.mean(qindex_map)

def mtf_resize(img, satellite='QuickBird', scale=4):
    # satellite GNyq
    scale = int(scale)
    if satellite == 'QuickBird':
        GNyq = [0.34, 0.32, 0.30, 0.22]  # Band Order: B,G,R,NIR
        GNyqPan = 0.15
    elif satellite == 'IKONOS':
        GNyq = [0.26, 0.28, 0.29, 0.28]  # Band Order: B,G,R,NIR
        GNyqPan = 0.17
    else:
        raise NotImplementedError('satellite: QuickBird or IKONOS')
    # lowpass
    img_ = img.squeeze()
    img_ = img_.astype(np.float64)
    if img_.ndim == 2:  # Pan
        H, W = img_.shape
        lowpass = GNyq2win(GNyqPan, scale, N=41)
    elif img_.ndim == 3:  # MS
        H, W, _ = img.shape
        lowpass = [GNyq2win(gnyq, scale, N=41) for gnyq in GNyq]
        lowpass = np.stack(lowpass, axis=-1)
    img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
    # downsampling
    output_size = (W // scale, H // scale)
    img_ = cv2.resize(img_, dsize=output_size, interpolation=cv2.INTER_NEAREST)
    return img_

def D_lambda(img_fake, img_lm, block_size=32, p=1):
    """Spectral distortion
    img_fake, generated HRMS
    img_lm, LRMS"""
    assert img_fake.ndim == img_lm.ndim == 3, 'Images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # D_lambda
    Q_fake = []
    Q_lm = []
    for i in range(C_f):
        for j in range(i + 1, C_f):
            # for fake
            band1 = img_fake[..., i]
            band2 = img_fake[..., j]
            Q_fake.append(_qindex(band1, band2, block_size=block_size))
            # for real
            band1 = img_lm[..., i]
            band2 = img_lm[..., j]
            Q_lm.append(_qindex(band1, band2, block_size=block_size))
    Q_fake = np.array(Q_fake)
    Q_lm = np.array(Q_lm)
    D_lambda_index = (np.abs(Q_fake - Q_lm) ** p).mean()
    return D_lambda_index ** (1 / p)

def D_s(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, q=1):
    """Spatial distortion
    img_fake, generated HRMS
    img_lm, LRMS
    pan, HRPan"""
    # fake and lm
    assert img_fake.ndim == img_lm.ndim == 3, 'MS images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert H_f // H_r == W_f // W_r == scale, 'Spatial resolution should be compatible with scale'
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # fake and pan
    assert pan.ndim == 3, 'Panchromatic image must be 3D!'
    H_p, W_p, C_p = pan.shape
    assert C_p == 1, 'size of 3rd dim of Panchromatic image must be 1'
    assert H_f == H_p and W_f == W_p, "Pan's and fake's spatial resolution should be the same"
    # get LRPan, 2D
    pan_lr = mtf_resize(pan, satellite=satellite, scale=scale)
    # print(pan_lr.shape)
    # D_s
    Q_hr = []
    Q_lr = []
    for i in range(C_f):
        # for HR fake
        band1 = img_fake[..., i]
        band2 = pan[..., 0]  # the input PAN is 3D with size=1 along 3rd dim
        # print(band1.shape)
        # print(band2.shape)
        Q_hr.append(_qindex(band1, band2, block_size=block_size))
        band1 = img_lm[..., i]
        band2 = pan_lr  # this is 2D
        # print(band1.shape)
        # print(band2.shape)
        Q_lr.append(_qindex(band1, band2, block_size=block_size))
    Q_hr = np.array(Q_hr)
    Q_lr = np.array(Q_lr)
    D_s_index = (np.abs(Q_hr - Q_lr) ** q).mean()
    return D_s_index ** (1 / q)


class AverageMeter_test_full(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = np.zeros([1,3])
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*np.array(n)
        self.count += n
        self.avg = self.sum / self.count


class Loss_test(nn.Module):
    def __init__(self, scale=8):
        super(Loss_test, self).__init__()
        self.scale=scale

    def forward(self, pred, hs, pan):
        # need H*W*C
        pred = np.clip(pred, 0, 1)
        pred = np.transpose(pred, [1,2,0])
        pan = np.transpose(pan, [1,2,0])
        hs = np.transpose(hs, [1,2,0])

        valid_error = np.zeros([1, 3])
        D_lambda_idx = D_lambda(pred, hs)
        D_s_idx = D_s(pred, hs, pan)
        QNR_idx = (1 - D_lambda_idx) * (1 - D_s_idx)

        valid_error[0, 0] = D_lambda_idx
        valid_error[0, 1] = D_s_idx
        valid_error[0, 2] = QNR_idx
        return valid_error



class AverageMeter_valid_full(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = np.zeros([1,3])
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*np.array(n)
        self.count += n
        self.avg = self.sum / self.count


class Loss_valid_full(nn.Module):
    def __init__(self, scale=4):
        super(Loss_valid_full, self).__init__()
        self.scale=scale

    def forward(self, pred, hs, pan):
        # need H*W*C
        pred = np.clip(pred, 0, 1)
        pred = np.transpose(pred, [1,2,0])
        pan = np.transpose(pan, [1,2,0])
        hs = np.transpose(hs, [1,2,0])

        valid_error = np.zeros([1, 3])
        D_lambda_idx = D_lambda(pred, hs)
        D_s_idx = D_s(pred, hs, pan)
        QNR_idx = (1 - D_lambda_idx) * (1 - D_s_idx)

        valid_error[0, 0] = D_lambda_idx
        valid_error[0, 1] = D_s_idx
        valid_error[0, 2] = QNR_idx
        return valid_error



