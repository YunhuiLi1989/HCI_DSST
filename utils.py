import scipy.io as sio   
import os   
import numpy as np   
import torch   
import logging   
import random   
from torch.utils.tensorboard import SummaryWriter   
import torchvision.utils as vutils   
import imgvision as iv   
from torchvision import transforms   
from ssim_torch import ssim   

 
def LoadTrainData(path):   
    imgs = []   
    scene_list = os.listdir(path)   
    scene_list.sort()   
    print('training scenes:', len(scene_list))   
    for i in range(len(scene_list)):   
        scene_path = path + scene_list[i]   
        scene_num = int(scene_list[i].split('.')[0][5:])   
        img_dict = sio.loadmat(scene_path)   
        img = img_dict['img_expand'] / 65536.   
        img = img.astype(np.float32)   
        img = np.transpose(img, (2, 0, 1))   
        img = torch.from_numpy(img)   
        imgs.append(img)   
        print('Training scene {} has been loaded. {}'.format(i+1, scene_list[i]))   
    return imgs   

def LoadValidData(path):   
    imgs = []   
    scene_list = os.listdir(path)   
    scene_list.sort()   
    print('testing scenes:', len(scene_list))   
    for i in range(len(scene_list)):   
        scene_path = path + scene_list[i]   
        scene_num = int(scene_list[i].split('.')[0][5:])   
        img_dict = sio.loadmat(scene_path)   
        img = img_dict['img'] / 1.   
        img = img.astype(np.float32)   
        img = np.transpose(img, (2, 0, 1))   
        img = torch.from_numpy(img)   
        imgs.append(img)   
        print('Testing scene {} has been loaded.'.format(i+1))   
    return imgs   

def LoadTrainMask(path, batch_size):   
    mask = sio.loadmat(path + '/mask.mat')   
    mask = mask['mask']   
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))   
    mask3d = np.transpose(mask3d, [2, 0, 1])   
    mask3d = torch.from_numpy(mask3d)   
    [nC, H, W] = mask3d.shape   
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()   

    [bs, nC, H, W] = mask3d_batch.shape   
    input_mask = torch.zeros(bs, nC, H, W + (nC-1)*2).cuda().float()   
    for i in range(nC):
        input_mask[:, i, :, 2 * i : 2 * i + W] = mask3d_batch[:, i, :, :]   

    return mask3d_batch, input_mask   

def LoadValidMask(path, valid_sam_num):   
    mask = sio.loadmat(path + '/mask.mat')   
    mask = mask['mask']   
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))   
    mask3d = np.transpose(mask3d, [2, 0, 1])   
    mask3d = torch.from_numpy(mask3d)   
    [nC, H, W] = mask3d.shape   
    mask3d_batch = mask3d.expand([valid_sam_num, nC, H, W]).cuda().float()   

    [bs, nC, H, W] = mask3d_batch.shape   
    input_mask = torch.zeros(bs, nC, H, W + (nC-1)*2).cuda().float()   
    for i in range(nC):
        input_mask[:, i, :, 2 * i : 2 * i + W] = mask3d_batch[:, i, :, :]   

    return mask3d_batch, input_mask   


def DataAugment(train_data, batch_size, crop_size=256, augment=True):   
    if augment:   
        data_batch = []   
         
        index = np.random.choice(range(len(train_data)), batch_size//2)   
        processed_data_1 = np.zeros((batch_size//2, 28, crop_size, crop_size), dtype=np.float32)   
        for i in range(batch_size//2):    
            img = train_data[index[i]]
            trans_crop = transforms.RandomCrop((crop_size, crop_size))   
            img = trans_crop(img)   
            rot_times = random.randint(0, 3)   
            for _ in range(rot_times):
                img = torch.rot90(img, dims=[1, 2])   
            trans_hflip = transforms.RandomHorizontalFlip(p=0.5)
            img = trans_hflip(img)   
            trans_vflip = transforms.RandomVerticalFlip(p=0.5)
            img = trans_vflip(img)   
            processed_data_1[i, :, :, :] = img   
        processed_data_1 = torch.from_numpy(processed_data_1).cuda()   
         
        processed_data_2 = np.zeros((batch_size-batch_size//2, 28, crop_size, crop_size), dtype=np.float32)   
        for i in range(batch_size-batch_size//2):   
            index_list = np.random.randint(0, len(train_data), 4)   
            img_list = []   
             
            for j in range(len(index_list)):
                img = train_data[index_list[j]]   
                trans_crop = transforms.RandomCrop((crop_size//2, crop_size//2))   
                img = trans_crop(img)   
                img_list.append(img)   
            processed_data_2[i, :, :crop_size//2, :crop_size//2] = img_list[0]   
            processed_data_2[i, :, :crop_size//2, crop_size//2:] = img_list[1]   
            processed_data_2[i, :, crop_size//2:, :crop_size//2] = img_list[2]   
            processed_data_2[i, :, crop_size//2:, crop_size//2:] = img_list[3]   
        processed_data_2 = torch.from_numpy(processed_data_2).cuda()   
        data_batch = torch.cat([processed_data_1, processed_data_2], dim=0)   
        return data_batch
    else:   
        data_batch = []   
        index = np.random.choice(range(len(train_data)), batch_size)   
        processed_data = np.zeros((batch_size, 28, crop_size, crop_size), dtype=np.float32)   
        for i in range(batch_size):          
            img = train_data[index[i]]   
            trans_crop = transforms.RandomCrop((crop_size, crop_size))   
            img = trans_crop(img)   
            processed_data[i, :, :, :] = img
        processed_data = torch.from_numpy(processed_data)   
        data_batch = processed_data   
        return data_batch

def DataGenerate(test_data, batch_size):   
    processed_data = np.zeros((batch_size, 28, 256, 256), dtype=np.float32)   
    for i in range(batch_size):
        img = test_data[i]   
        processed_data[i, :, :, :] = img
    processed_data = torch.from_numpy(processed_data)   
    data_batch = processed_data   
    return data_batch

def GenerateMeasurement(data_batch, mask3d_batch):   
    temp = mask3d_batch * data_batch   
    [bs, nC, H, W] = temp.shape
    output = torch.zeros(bs, nC, H , W + (nC-1)*2).cuda()   
    for i in range(nC):
        output[:, i, :, 2*i:2*i + W] = temp[:, i, :, :]   
    meas = torch.sum(output, 1)   
    meas = meas / nC * 2   
    [bs, H, W] = meas.shape   
    output = torch.zeros(bs, nC, H , W - (nC-1)*2).cuda()   
    for i in range(nC):
        output[:, i, :, :] = meas[:, :, 2*i:2*i + W - (nC-1)*2]
    H3 = torch.mul(output, mask3d_batch).cuda()   
    H1 = output.cuda()   
    H2 = mask3d_batch.cuda()   
     
    return torch.cat([H1, H2], dim=1)


def LoadMeasurement(path):   
    file = os.listdir(path)   
    file_path = path + file[1]   
    meas = sio.loadmat(file_path)    
    meas = meas.astype(np.float32)   
    meas = torch.from_numpy(meas)   
    return meas   
 

def Seed_Torch(seed=1029):   
    random.seed(seed)   
    os.environ['PYTHONHASHSEED'] = str(seed)   
    np.random.seed(seed)   
    torch.manual_seed(seed)   
    torch.cuda.manual_seed(seed)   
    torch.cuda.manual_seed_all(seed)   
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

 
def Quality_Eval(img, ref):   
    np.seterr(divide='ignore', invalid='ignore')   
    metric = iv.spectra_metric(img.permute(1, 2, 0).detach().cpu().numpy(), ref.permute(1, 2, 0).detach().cpu().numpy(), 1)   
    psnr = metric.PSNR()
    ssim = metric.SSIM()
    sam = metric.SAM()
    return psnr, ssim, sam   

def Torch_Psnr(img, ref):   
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def Torch_Ssim(img, ref):   
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

 
def tensorboard_show(writer, loss_avg, psnr_mean, ssim_mean, epoch, truth, recon):
    writer.add_scalar('result/train_loss_avg', loss_avg, epoch)   
    writer.add_scalar('result/valid_psnr_avg', psnr_mean, epoch)   
    writer.add_scalar('result/valid_ssim_avg', ssim_mean, epoch)   
    truth_spectral = np.expand_dims(truth[5, :, :, :],1).repeat(3, 1)   
    recon_spectral = np.expand_dims(recon[5, :, :, :], 1).repeat(3, 1)   
    map_RGB = np.array([[33, 0, 255], [0, 0, 255], [0, 66, 255], [0, 99, 255], [0, 140, 255], [0, 163, 247], [0, 183, 214],   
              [0, 201, 182], [0, 218, 146], [0, 234, 116], [0, 251, 78], [0, 255, 0], [0, 255, 0], [0, 255, 0],
              [0, 255, 0], [0, 255, 0], [0, 255, 0], [148, 255, 0], [217, 255, 0], [255, 254, 0], [255, 227, 0],
              [255, 189, 0], [255, 137, 0], [255, 77, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]], dtype=np.float32)   
    map_RGB = np.expand_dims(map_RGB, 2).repeat(256,2)   
    map_RGB = np.expand_dims(map_RGB, 3).repeat(256, 3)   
    truth_spectral = np.multiply(truth_spectral, map_RGB) / 256   
    recon_spectral = np.multiply(recon_spectral, map_RGB) / 256   
    truth_spectral = vutils.make_grid(torch.tensor(truth_spectral), 7, 20)   
    recon_spectral = vutils.make_grid(torch.tensor(recon_spectral), 7, 20)   
    writer.add_image('image/truth', truth_spectral, epoch, dataformats='CHW')   
    writer.add_image('image/recon', recon_spectral, epoch, dataformats='CHW')   

 
def time2file_name(time):   
    year = time[0:4]   
    month = time[5:7]   
    day = time[8:10]   
    hour = time[11:13]   
    minute = time[14:16]   
    second = time[17:19]   
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename   

def gen_log(path):   
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)   
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")   

    log_file = path + '/log.txt'   
    fh = logging.FileHandler(log_file, mode='a')   
    fh.setLevel(logging.INFO)   
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()  
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)   
    logger.addHandler(ch)
    return logger

def checkpoint_save(model, optimizer, scheduler, epoch, model_path, logger):
    model_out_path = model_path + "model_epoch_{}.pth".format(epoch)
    torch.save({'epoch': epoch,   
                'model_dict': model.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict()
                }, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))   

def checkpoint_load(model, optimizer, scheduler, epoch, model_path, logger):
    model_load_path = model_path + "model_epoch_{}.pth".format(epoch)
    checkpoint = torch.load(model_load_path, map_location=torch.device('cuda:0'))   
    model.load_state_dict(checkpoint['model_dict'], strict=False)   
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_dict'])
    logger.info("Checkpoint loaded from {}".format(model_load_path))   