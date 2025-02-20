import argparse   

parser = argparse.ArgumentParser(description="Hyperspectral Image Reconstruction ProjectA")   

 
parser.add_argument("--gpu_id", type=str, default='0')   

 
parser.add_argument('--train_data_path', type=str, default='./dataset/train_data/', help='train data directory')   
parser.add_argument('--mask_data_path', type=str, default='./dataset/', help='mask data directory')   
parser.add_argument('--valid_data_path', type=str, default='./dataset/valid_data/', help='valid data directory')   
parser.add_argument('--test_data_path', type=str, default='./dataset/test_data/', help='test data directory')   

 
parser.add_argument('--output_path', type=str, default='./output/', help='saving_path')   

 
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')   

 
parser.add_argument('--batch_size', type=int, default=3, help='the number of HSIs per batch')   
parser.add_argument("--max_epoch", type=int, default=4000, help='total epoch')   
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')   
parser.add_argument("--milestones", type=int, default=[500, 1000, 1500, 2000, 2500], help='milestones for MultiStepLR')   
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')   
parser.add_argument("--epoch_sam_num", type=int, default=500, help='the number of samples per epoch')   
parser.add_argument("--learning_rate", type=float, default=0.0004)   

 
parser.add_argument("--valid_sam_num", type=int, default=10, help='the number of samples for validation')   

 
parser.add_argument("--test_sam_num", type=int, default=10, help='the number of samples for testing')   

opt = parser.parse_args()   

for arg in vars(opt):           
    if vars(opt)[arg] == 'True':   
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False