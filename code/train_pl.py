import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from networks.vnet import VNet_MTPD
from networks.ResNet34 import Resnet341
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, ToTensor, TwoStreamBatchSampler
# from dataloaders.dataset import *
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--exp', type=str,  default="pl19", help='model_name')                               # todo model name
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
# PD
parser.add_argument('--uncertainty_th', type=float,  default=0.1, help='threshold')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)
#patch_size = (96, 96, 96)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)
def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

# def getPrototype(features, mask, class_confidence):
#     fts = F.interpolate(features, size=mask.shape[-3:], mode='trilinear')  
#     mask_new = mask.unsqueeze(1) 
#     #masked_features = features*mask_new
#     masked_features = torch.mul(fts, mask_new)  
#     masked_fts = torch.sum(masked_features*class_confidence, dim=(2, 3, 4)) / ((mask_new*class_confidence).sum(dim=(2, 3, 4)) + 1e-5)  # bs x C
#     return masked_fts
def getPrototype(features, mask, class_confidence):
    #fts = F.interpolate(features, size=mask.shape[-3:], mode='trilinear')  
    mask_new = mask.unsqueeze(1) 
    #masked_features = features*mask_new
    masked_features = torch.mul(features, mask_new)  
    masked_fts = torch.sum(masked_features*class_confidence, dim=(2, 3, 4)) / ((mask_new*class_confidence).sum(dim=(2, 3, 4)) + 1e-5)  # bs x C
    return masked_fts
def calDist(fts,  prototype):

    #fts_adj_size = F.interpolate(fts, size=mask.shape[-3:], mode='trilinear')
    prototype_new = prototype.unsqueeze(2)
    prototype_new = prototype_new.unsqueeze(3)
    prototype_new = prototype_new.unsqueeze(4)
    dist = torch.sum(torch.pow(fts - prototype_new, 2), dim=1, keepdim=True)
    return dist
# def calDist(fts, mask, prototype):

#     fts_adj_size = F.interpolate(fts, size=mask.shape[-3:], mode='trilinear')
#     prototype_new = prototype.unsqueeze(2)
#     prototype_new = prototype_new.unsqueeze(3)
#     prototype_new = prototype_new.unsqueeze(4)
#     dist = torch.sum(torch.pow(fts_adj_size - prototype_new, 2), dim=1, keepdim=True)
#     return dist
if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(name ='vnet'):
        # Network definition
        if name == 'vnet':
            net = VNet_MTPD(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        if name == 'resnet34':
            net = Resnet341(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        return model

    model_vnet = create_model(name='vnet')
    model_resnet = create_model(name='resnet34')

    db_train = LAHeart(base_dir=train_data_path,
                               split='train',
                               train_flod='train0.list',                   # todo change training flod
                               common_transform=transforms.Compose([
                                   RandomCrop(patch_size),
                               ]),
                               sp_transform=transforms.Compose([
                                   ToTensor(),
                               ]))
    labeled_idxs = list(range(16))           # todo set labeled num
    unlabeled_idxs = list(range(16, 80))     # todo set labeled num all_sample_num

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    vnet_optimizer = optim.SGD(model_vnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    resnet_optimizer = optim.SGD(model_resnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0

    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model_vnet.train()
    model_resnet.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            print('epoch:{}'.format(epoch_num))
            volume_batch1, volume_label1 = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch2, volume_label2 = sampled_batch[1]['image'], sampled_batch[1]['label']

            v_input,v_label = volume_batch1.cuda(), volume_label1.cuda()
            r_input,r_label = volume_batch2.cuda(), volume_label2.cuda()

            v_outputs = model_vnet  (v_input)
            r_outputs = model_resnet(r_input)
                        
            v_soft = F.softmax(v_outputs, dim=1)
            r_soft = F.softmax(r_outputs, dim=1)
            # noise = torch.clamp(torch.randn_like(v_input[labeled_bs:]) * 0.1, -0.2, 0.2)
            # noi_v_input = noise+v_input[labeled_bs:]
            # noi_r_input = noise+r_input[labeled_bs:]
            # mean_v_soft=torch.mean(v_soft[labeled_bs:])
            # mean_r_soft=torch.mean(r_soft[labeled_bs:])
            
            # vsoft_mask = (v_soft[labeled_bs:]>=mean_v_soft).long()
            # vsoft_mask = vsoft_mask.detach().cpu().numpy()
            # # vsoft_mask2 = (v_soft[labeled_bs:]<mean_v_soft).long()
            # # vsoft_mask2 =vsoft_mask2.detach().cpu().numpy()

            # rsoft_mask = (r_soft[labeled_bs:]>=mean_r_soft).long()
            # rsoft_mask = rsoft_mask.detach().cpu().numpy()
            # rsoft_mask2 = (r_soft[labeled_bs:]<mean_r_soft).long()
            # rsoft_mask2 =rsoft_mask2.detach().cpu().numpy()           

            v_oh   = torch.argmax(v_soft, dim=1)
            r_oh   = torch.argmax(r_soft, dim=1)
            v_oh = v_oh.detach().cpu().numpy()
            r_oh = r_oh.detach().cpu().numpy()

            vone_hot_labels = []
            rone_hot_labels = []
            v_std_box = []
            r_std_box = []
            v_std_box2 = []
            r_std_box2 = []
            
            same_mask =(v_oh[labeled_bs:]==r_oh[labeled_bs:])
            diff_mask =(same_mask==0)
            
            for i in range(3): 
                with torch.no_grad():                   
                    v_output1= model_vnet  (v_input[labeled_bs:])#noi_v_input
                    #vf1= model_vnet.featuremap_center
                    v_soft2= F.softmax(v_output1, dim=1)
                    v_onehot= torch.argmax(v_soft2, dim=1)
                    v_onehot=v_onehot.detach().cpu().numpy()
                    vone_hot_labels.append(v_onehot)
                    #r_output1= model_resnet(noi_r_input)
                    r_output1= model_resnet  (r_input[labeled_bs:])#noi_v_input  noi_r_input
                    #rf1= model_resnet.featuremap_center
                    r_soft2= F.softmax(r_output1, dim=1)
                    r_onehot= torch.argmax(r_soft2, dim=1) 
                    r_onehot= r_onehot.detach().cpu().numpy()                   
                    rone_hot_labels.append(r_onehot)
                    
                    # v_std_box.append(v_output1)
                    # r_std_box.append(r_output1)
            
                    # v_std_box2.append(v_output1)
                    # r_std_box2.append(rf1)
            # vpreds2   = torch.stack(v_std_box)
            # rpreds2   = torch.stack(r_std_box)

            # # vf2 = torch.stack(v_std_box2)
            # # rf2 = torch.stack(r_std_box2)
            # vf_mean =torch.mean(vpreds2,dim=0)
            # rf_mean =torch.mean(rpreds2,dim=0)
            # #print(vpreds2.shape)
            # vpreds2   = torch.sigmoid(vpreds2/2.0)
            # v_std2    = torch.std(vpreds2,dim=0).detach().cpu().numpy()
            # mean_v_std=np.mean(v_std2)
            
            # rpreds2   = torch.sigmoid(rpreds2/2.0)
            # r_std2    = torch.std(rpreds2,dim=0).detach().cpu().numpy()
            # mean_r_std=np.mean(r_std2)                    

            # vkl_div2 = F.kl_div(v_soft[labeled_bs:].log(),r_soft[labeled_bs:],reduction='none')
            # vkl_div2 = torch.sum(vkl_div2, dim=1).detach().cpu().numpy()
            # mean_vkl_div=np.mean(vkl_div2)               
                                     
            # rkl_div2 = F.kl_div(r_soft[labeled_bs:].log(),v_soft[labeled_bs:],reduction='none')
            # rkl_div2 = torch.sum(rkl_div2, dim=1).detach().cpu().numpy()
            # mean_rkl_div=np.mean(rkl_div2)   
                                   
            # #print('mean_kl_div,mean_std ',mean_kl_div,mean_std)
            # mean_v_std_mask = (v_std2 < mean_v_std)
            # mean_r_std_mask = (r_std2 < mean_r_std)
            # mean_v_div      = (vkl_div2 < mean_vkl_div)
            # mean_r_div      = (rkl_div2 < mean_rkl_div)
            
            #voh_mask =((v_oh[labeled_bs:]==vone_hot_labels[0])&(vone_hot_labels[0]==vone_hot_labels[1])&(vone_hot_labels[0]==vone_hot_labels[2]))
            voh_mask =(((v_oh[labeled_bs:]==vone_hot_labels[0])&(vone_hot_labels[0]==rone_hot_labels[0]))|\
                       ((v_oh[labeled_bs:]==vone_hot_labels[1])&(vone_hot_labels[1]==rone_hot_labels[1]))|\
                       ((v_oh[labeled_bs:]==vone_hot_labels[2])&(vone_hot_labels[2]==rone_hot_labels[2]))
                        )
            #voh_mask3  = ((v_oh==vone_hot_labels[0])|(v_oh==vone_hot_labels[1])|(v_oh==vone_hot_labels[2]))
            #voh_mask =voh_mask.detach().cpu().numpy()
            voh_mask2 =(((v_oh[labeled_bs:]==vone_hot_labels[0])&(vone_hot_labels[0]==vone_hot_labels[1])&(vone_hot_labels[0]==vone_hot_labels[2])&\
                        (vone_hot_labels[0]==rone_hot_labels[0])&(rone_hot_labels[0]==rone_hot_labels[1])&(rone_hot_labels[2]==rone_hot_labels[1])))
            #voh_mask2 =voh_mask2.detach().cpu().numpy()

            #roh_mask =((r_oh[labeled_bs:]==rone_hot_labels[0])&(rone_hot_labels[0]==rone_hot_labels[1])&(rone_hot_labels[0]==rone_hot_labels[2]))
            roh_mask =(((r_oh[labeled_bs:]==rone_hot_labels[0])&(vone_hot_labels[0]==rone_hot_labels[0]))|\
                       ((r_oh[labeled_bs:]==rone_hot_labels[1])&(vone_hot_labels[1]==rone_hot_labels[1]))|\
                       ((r_oh[labeled_bs:]==rone_hot_labels[2])&(vone_hot_labels[2]==rone_hot_labels[2]))
                        )
           # roh_mask3  = ((r_oh==rone_hot_labels[0])|(r_oh==rone_hot_labels[1])|(r_oh==rone_hot_labels[2]))
            #roh_mask =roh_mask.detach().cpu().numpy()
            roh_mask2 =(((r_oh[labeled_bs:]==rone_hot_labels[0])&(rone_hot_labels[0]==rone_hot_labels[1])&(rone_hot_labels[0]==rone_hot_labels[2])&\
                        (rone_hot_labels[0]==vone_hot_labels[0])&(vone_hot_labels[0]==vone_hot_labels[1])&(vone_hot_labels[2]==vone_hot_labels[1])))
            #roh_mask2 =roh_mask2.detach().cpu().numpy()
            # v_oh= torch.from_numpy(v_oh).cuda()
            # r_oh= torch.from_numpy(r_oh).cuda()
            ##################################################
            # un_v_high_mask  = mean_v_std_mask*mean_v_div*same_mask#*voh_mask
            # un_r_high_mask  = mean_r_std_mask*mean_r_div*same_mask#*roh_mask
            
            # vnoisy_mask     = (un_v_high_mask ==0)
            # rnoisy_mask     = (un_r_high_mask ==0)
            
            # un_v_high_only_mask = (mean_v_std_mask*mean_v_div*same_mask   )|(mean_v_std_mask*mean_v_div*diff_mask*voh_mask2)
            # un_r_high_only_mask = (mean_r_std_mask*mean_r_div*same_mask   )|(mean_r_std_mask*mean_r_div*diff_mask*roh_mask2)
            # un_v_high_only_mask = torch.from_numpy(un_v_high_only_mask).long().cuda()          
            # un_r_high_only_mask = torch.from_numpy(un_r_high_only_mask).long().cuda() 

            un_v_pro_mask       = (same_mask*voh_mask)|(diff_mask*voh_mask2)
            un_r_pro_mask       = (same_mask*roh_mask)|(diff_mask*roh_mask2)        
            un_v_pro_mask       = torch.from_numpy(un_v_pro_mask).long().cuda()          
            un_r_pro_mask       = torch.from_numpy(un_r_pro_mask).long().cuda() 

            # vone_hot_labels     = torch.from_numpy(vone_hot_labels).long().cuda() 
            # rone_hot_labels     = torch.from_numpy(rone_hot_labels).long().cuda()
        #################################################################################   
            vrect_output_soft   = un_v_pro_mask * v_soft[args.labeled_bs:]#high 
            vrect_output_onehot = torch.argmax(vrect_output_soft, dim=1)
            vobj_confi = v_soft[args.labeled_bs:, 1, ...].unsqueeze(1)
            vobj_prototype = getPrototype(v_outputs[labeled_bs:], vrect_output_onehot, vobj_confi) 
            vdistance_f_obj= calDist(v_outputs[labeled_bs:],vobj_prototype) 
            #print('vdistance_f_obj ',vdistance_f_obj)
            vbg_confi  = v_soft[args.labeled_bs:, 0, ...].unsqueeze(1)
            vrect_bg_output_onehot= (vrect_output_onehot == 0)
            vbg_prototype  = getPrototype(v_outputs[labeled_bs:], vrect_bg_output_onehot, vbg_confi) 
            vdistance_f_bg = calDist(v_outputs[labeled_bs:], vbg_prototype) 
                       
            selection_mask_bg  = torch.zeros(vdistance_f_bg.shape).cuda()
            selection_mask_obj = torch.zeros(vdistance_f_obj.shape).cuda()    

            selection_mask_bg [vdistance_f_obj-vdistance_f_bg>0] = 1.0
            selection_mask_obj[vdistance_f_bg-vdistance_f_obj>0] = 1.0
            vrec_obj_label = selection_mask_obj.squeeze(1)
            vrec_obj_label = vrec_obj_label.detach().cpu().numpy()
            # diff_bg_greater_obj = vdistance_f_obj- vdistance_f_bg
            # diff_bg_greater_obj = diff_bg_greater_obj *un_v_high_mask[:,0,...]
            # diff_obj_greater_bg = vdistance_f_bg - vdistance_f_obj
            # diff_obj_greater_bg = diff_obj_greater_bg *un_v_high_mask[:,1,...]
        #######################################################################################    
            rrect_output_soft = un_r_pro_mask * r_soft[args.labeled_bs:]
            rrect_output_onehot = torch.argmax(rrect_output_soft, dim=1)
            robj_confi = r_soft[args.labeled_bs:, 1, ...].unsqueeze(1)
            robj_prototype = getPrototype(r_outputs[labeled_bs:], rrect_output_onehot, robj_confi) 
            rdistance_f_obj= calDist(r_outputs[labeled_bs:],robj_prototype) 
            
            rbg_confi  = r_soft[args.labeled_bs:, 0, ...].unsqueeze(1)
            rrect_bg_output_onehot= (rrect_output_onehot == 0)
            rbg_prototype = getPrototype(r_outputs[labeled_bs:], rrect_bg_output_onehot, rbg_confi) 
            rdistance_f_bg = calDist(r_outputs[labeled_bs:],rbg_prototype) 
                       
            rselection_mask_bg  = torch.zeros(rdistance_f_bg.shape).cuda()
            rselection_mask_obj = torch.zeros(rdistance_f_obj.shape).cuda()           

            rselection_mask_bg [rdistance_f_obj-rdistance_f_bg >0] = 1.0
            rselection_mask_obj[rdistance_f_bg -rdistance_f_obj>0] = 1.0
            rrec_obj_label = rselection_mask_obj.squeeze(1)
            rrec_obj_label = rrec_obj_label.detach().cpu().numpy()
            rselection_mask = torch.cat((rselection_mask_bg, rselection_mask_obj), dim=1)
            selection_mask  = torch.cat((selection_mask_bg, selection_mask_obj), dim=1) 
                       
            #unlabel_v   = torch.zeros_like(r_outputs[args.labeled_bs:]).scatter_(dim=1,index= v_oh[args.labeled_bs:].unsqueeze(dim=1),src=torch.ones_like(r_outputs[args.labeled_bs:])).cuda() 
            #rec_v_label = torch.zeros_like(r_outputs[args.labeled_bs:]).scatter_(dim=1,index= vrec_obj_label.long().unsqueeze(dim=1),src=torch.ones_like(r_outputs[args.labeled_bs:])).cuda() 
            vclean_mask = np.zeros([r_oh[args.labeled_bs:].shape[0],  r_oh[args.labeled_bs:].shape[1],  r_oh[args.labeled_bs:].shape[2],  r_oh[args.labeled_bs:].shape[3]])                       
            #vclean_mask2 = torch.zeros([r_oh[args.labeled_bs:].shape[0], 2, r_oh[args.labeled_bs:].shape[1],  r_oh[args.labeled_bs:].shape[2],  r_oh[args.labeled_bs:].shape[3]]).cuda()                        

            #unlabel_r   = torch.zeros_like(r_outputs[args.labeled_bs:]).scatter_(dim=1,index= r_oh[args.labeled_bs:].unsqueeze(dim=1),src=torch.ones_like(r_outputs[args.labeled_bs:])).cuda() 
            #rec_r_label = torch.zeros_like(r_outputs[args.labeled_bs:]).scatter_(dim=1,index= rrec_obj_label.long() .unsqueeze(dim=1),src=torch.ones_like(r_outputs[args.labeled_bs:])).cuda() 
            rclean_mask = np.zeros([v_oh[args.labeled_bs:].shape[0],  v_oh[args.labeled_bs:].shape[1],  v_oh[args.labeled_bs:].shape[2],  v_oh[args.labeled_bs:].shape[3]])
            #rclean_mask2 = torch.zeros([v_oh[args.labeled_bs:].shape[0], 2, v_oh[args.labeled_bs:].shape[1],  v_oh[args.labeled_bs:].shape[2],  v_oh[args.labeled_bs:].shape[3]]).cuda() 

            vclean_mask[((vrec_obj_label==1)|(rrec_obj_label==1))&(((vone_hot_labels[0]==1)&(rone_hot_labels[0]==1))|\
                                                                   ((vone_hot_labels[1]==1)&(rone_hot_labels[1]==1))|\
                                                                   ((vone_hot_labels[2]==1)&(rone_hot_labels[2]==1)))|\
                        ((rone_hot_labels[0]==1)&(rone_hot_labels[1]==1)&(rone_hot_labels[2]==1))] = 1.0
            rclean_mask[((vrec_obj_label==1)|(rrec_obj_label==1))&(((vone_hot_labels[0]==1)&(rone_hot_labels[0]==1))|\
                                                                   ((vone_hot_labels[1]==1)&(rone_hot_labels[1]==1))|\
                                                                   ((vone_hot_labels[2]==1)&(rone_hot_labels[2]==1)))|\
                        ((vone_hot_labels[0]==1)&(vone_hot_labels[1]==1)&(vone_hot_labels[2]==1))] = 1.0

            vmask1 = ((((vrec_obj_label==1)|(rrec_obj_label==1))&(((vone_hot_labels[0]==1)&(rone_hot_labels[0]==1))|((vone_hot_labels[1]==1)&(rone_hot_labels[1]==1))|((vone_hot_labels[2]==1)&(rone_hot_labels[2]==1))))|\
                      ((rone_hot_labels[0]==1)&(rone_hot_labels[1]==1)&(rone_hot_labels[2]==1))|\
                      (((vrec_obj_label==0)|(rrec_obj_label==0))&(((vone_hot_labels[0]==0)&(rone_hot_labels[0]==0))|((vone_hot_labels[1]==0)&(rone_hot_labels[1]==0))|((vone_hot_labels[2]==0)&(rone_hot_labels[2]==0))))|\
                      ((rone_hot_labels[0]==0)&(rone_hot_labels[1]==0)&(rone_hot_labels[2]==0))\
                        )
                      
            rmask1 = ((((vrec_obj_label==1)|(rrec_obj_label==1))&(((vone_hot_labels[0]==1)&(rone_hot_labels[0]==1))|((vone_hot_labels[1]==1)&(rone_hot_labels[1]==1))|((vone_hot_labels[2]==1)&(rone_hot_labels[2]==1))))|\
                     ((vone_hot_labels[0]==1)&(vone_hot_labels[1]==1)&(vone_hot_labels[2]==1)) |\
                      (((vrec_obj_label==0)|(rrec_obj_label==0))&(((vone_hot_labels[0]==0)&(rone_hot_labels[0]==0))|((vone_hot_labels[1]==0)&(rone_hot_labels[1]==0))|((vone_hot_labels[2]==0)&(rone_hot_labels[2]==0))))|\
                      ((vone_hot_labels[0]==0)&(vone_hot_labels[1]==0)&(vone_hot_labels[2]==0))\
                        )
            vmask1= torch.from_numpy(vmask1).long().cuda()
            rmask1= torch.from_numpy(rmask1).long().cuda()

            v_outputs2 = vmask1*v_outputs[args.labeled_bs:]
            r_outputs2 = rmask1*r_outputs[args.labeled_bs:]
           ####################################################################
            consistency_weight = get_current_consistency_weight(iter_num//150)
            vclean_mask= torch.from_numpy(vclean_mask).long().cuda()
            rclean_mask= torch.from_numpy(rclean_mask).long().cuda()
            un_v_loss         = F.cross_entropy(v_outputs2, vclean_mask)
            un_v_loss         = consistency_weight * un_v_loss

            un_r_loss         = F.cross_entropy(r_outputs2, rclean_mask)
            un_r_loss         = consistency_weight * un_r_loss
####################################################################################################################################           
            ## calculate the supervised loss
            v_loss_seg = F.cross_entropy(v_outputs[:labeled_bs], v_label[:labeled_bs])
            v_outputs_soft = F.softmax(v_outputs, dim=1)            
            v_loss_seg_dice = losses.dice_loss(v_outputs_soft[:labeled_bs, 1, :, :, :], v_label[:labeled_bs] == 1)
            r_loss_seg = F.cross_entropy(r_outputs[:labeled_bs], r_label[:labeled_bs])
            r_outputs_soft = F.softmax(r_outputs, dim=1)
            r_loss_seg_dice = losses.dice_loss(r_outputs_soft[:labeled_bs, 1, :, :, :], r_label[:labeled_bs] == 1)           
            v_supervised_loss =  (v_loss_seg + v_loss_seg_dice) 
            r_supervised_loss =  (r_loss_seg + r_loss_seg_dice)            
           #################################################################
            v_loss = v_supervised_loss+r_supervised_loss+un_v_loss+un_r_loss
         ############uncertainty minimize              
            vnet_optimizer.zero_grad()
            resnet_optimizer.zero_grad()
            v_loss.backward()
            #r_loss.backward()
            vnet_optimizer.step()
            resnet_optimizer.step()
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/v_loss', v_loss, iter_num)
            writer.add_scalar('loss/v_loss_seg', v_loss_seg, iter_num)
            writer.add_scalar('loss/v_loss_seg_dice', v_loss_seg_dice, iter_num)
            writer.add_scalar('loss/v_supervised_loss', v_supervised_loss, iter_num)
            writer.add_scalar('loss/r_loss_seg', r_loss_seg, iter_num)
            writer.add_scalar('loss/r_loss_seg_dice', r_loss_seg_dice, iter_num)
            writer.add_scalar('loss/r_supervised_loss', r_supervised_loss, iter_num)

            logging.info(
                'iteration : %d v_supervised_loss : %f   v_loss:%.4f '  %
                (iter_num,
                 v_supervised_loss.item(),v_loss.item()                                
                 ))
            if iter_num % 6000 == 0 and iter_num!= 0:
                save_mode_path_vnet = os.path.join(snapshot_path, 'vnet_iter_' + str(iter_num) + '.pth')
                torch.save(model_vnet.state_dict(), save_mode_path_vnet)
                logging.info("save model to {}".format(save_mode_path_vnet))

                save_mode_path_resnet = os.path.join(snapshot_path, 'resnet_iter_' + str(iter_num) + '.pth')
                torch.save(model_resnet.state_dict(), save_mode_path_resnet)
                logging.info("save model to {}".format(save_mode_path_resnet))
            ## change lr
            if iter_num % 2500 == 0 and iter_num!= 0:
                lr_ = lr_ * 0.1
                for param_group in vnet_optimizer.param_groups:
                    param_group['lr'] = lr_
                for param_group in resnet_optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num >= max_iterations:
                break
            time1 = time.time()

            iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    save_mode_path_vnet = os.path.join(snapshot_path, 'vnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_vnet.state_dict(), save_mode_path_vnet)
    logging.info("save model to {}".format(save_mode_path_vnet))

    save_mode_path_resnet = os.path.join(snapshot_path, 'resnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_resnet.state_dict(), save_mode_path_resnet)
    logging.info("save model to {}".format(save_mode_path_resnet))

    writer.close()
