import numpy

from utils import *   
from DSST_M import *
import torch   
import scipy.io as scio    
import time   
import os   
import numpy as np   
from torch.autograd import Variable   
import datetime   
from option import opt   
from thop import profile   
from torchsummary import summary   
from fvcore.nn import FlopCountAnalysis, parameter_count_table   

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'   
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id   
torch.backends.cudnn.enabled = True   
torch.backends.cudnn.benchmark = True   
if not torch.cuda.is_available():   
    raise Exception('NO GPU!')

 
 
seed = 12
Seed_Torch(seed)   

 
mask3d_test, input_mask_test = LoadValidMask(opt.mask_data_path, opt.test_sam_num)   

 
test_data = LoadValidData(opt.test_data_path)   

 
date_time = str(datetime.datetime.now())   
date_time = time2file_name(date_time)   
result_path = opt.output_path + date_time + '/result/'   
model_path = opt.output_path + date_time + '/model/'   
tensorboard_path = opt.output_path + date_time + '/tensorboard/'   
if not os.path.exists(result_path):
    os.makedirs(result_path)   
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)   
writer = SummaryWriter(log_dir=tensorboard_path)   


 
model = Run().cuda()   
 
input_x_rand = torch.randn(1, 2 * 28, 256, 256).cuda()   
flops, params = profile(model, inputs=[input_x_rand])   
print('flops:{}'.format(flops/(1024*1024*1024)))   
print('params:{}'.format(params/(1024*1024)))   
 
flops = FlopCountAnalysis(model, inputs=tuple([input_x_rand]))   
print('flops:{}'.format(flops.total()/(1024*1024*1024)))   
 
n_param = sum([p.nelement() for p in model.parameters()])   
print('params:{}'.format(n_param/(1024*1024)))   

 
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))   
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)   
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
mse = torch.nn.MSELoss().cuda()   

 
resume = False   


def test(epoch, logger):   
    psnr_list = []   
    ssim_list = []   

    data_test = DataGenerate(test_data, opt.test_sam_num)   
    test_truth = data_test.cuda().float()   
    meas_test = GenerateMeasurement(test_truth, mask3d_test)   
    model.eval()   
    begin_time = time.time()   
    with torch.no_grad():
        model_out = model(meas_test)   
    end_time = time.time()   
    for i in range(test_truth.shape[0]):
        psnr_val = Torch_Psnr(model_out[i, :, :, :], test_truth[i, :, :, :])
        ssim_val = Torch_Ssim(model_out[i, :, :, :], test_truth[i, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())

    recon = model_out.detach().cpu().numpy().astype(np.float32)   
    truth = test_truth.cpu().numpy().astype(np.float32)   
    psnr_mean = np.mean(np.asarray(psnr_list))   
    ssim_mean = np.mean(np.asarray(ssim_list))   

    logger.info('===> Epoch {}: test psnr = {:.2f}, test ssim = {:.4f}, time: {:.4f}'.format(epoch, psnr_mean, ssim_mean, (end_time - begin_time)))   
    model.train()   
    return recon, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    logger = gen_log(model_path)   
    logger.info("Learning rate:{}, batch_size:{}, seed:{}.\n".format(opt.learning_rate, opt.batch_size, seed))   

    model_load_path = './model/'   
    epoch = 'dsst_m'
    checkpoint_load(model, optimizer, scheduler, epoch, model_load_path, logger)   
    loss_avg = 0
    (recon, truth, psnr_list, ssim_list, psnr_mean, ssim_mean) = test(epoch, logger)
    '''
    tensorboard_show(writer, loss_avg, psnr_mean, ssim_mean, epoch, truth, recon)   
    scheduler.step()
    if (epoch % 100 == 0) or (epoch >= 3900):   
        name = result_path + '/' + 'Test_{}_{:.2f}_{:.4f}'.format(epoch, psnr_mean, ssim_mean) + '.mat'   
        scio.savemat(name, {'truth': truth, 'recon': recon, 'psnr_list': psnr_list, 'ssim_list': ssim_list})
        checkpoint_save(model, optimizer, scheduler, epoch, model_path, logger)   
    
    writer.close()   
    '''

if __name__ == '__main__':   
    main()    


