import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np


import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file

# """
class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        # 设置随机数种子
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            np.random.seed(int(cfg.SEED))
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)
            """
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            """

        # 单机多卡
        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda")

        # SCST标记
        self.rl_stage = False
        # 设置日志写入
        self.setup_logging()
        # 训练数据集
        self.setup_dataset()
        # 训练模型结构
        self.setup_network()
        # 模型验证
        self.val_evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.VAL_ID,  # 图像id文件  './mscoco/txt/coco_val_image_id.txt'
            gv_feat = cfg.DATA_LOADER.VAL_GV_FEAT,
            att_feats = cfg.DATA_LOADER.VAL_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.VAL_ANNFILE
        )
        self.test_evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.TEST_ID,  # 图像id文件  './mscoco/txt/coco_test_image_id.txt'
            gv_feat = cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE
        )
        self.scorer = Scorer()

        self.scaler = torch.cuda.amp.GradScaler()

    # 设置日志写入
    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        # 使用多卡训练时不输出日志
        if self.distributed and dist.get_rank() > 0:
            return

        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        
        """
        # 日志的屏幕打印
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        """

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)

        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
        # 模型构建
        model = models.create(cfg.MODEL.TYPE)
        print(model)

        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device),
                device_ids = [self.args.local_rank],
                output_device = self.args.local_rank,
                broadcast_buffers = False
            )
        else:
            self.model = torch.nn.DataParallel(model).cuda()

        # 如果resume > 0，则需要导入参数
        # 此处导入参数到CPU上？
        if self.args.resume > 0:
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model", self.args.resume),
                    map_location=lambda storage, loc: storage)
            )

        # 判断是否导入epoch
        # self.load_epoch = -1
        self.load_epoch = self.args.resume - 1 # @@@@
        self.load_iteration = -1

        if self.args.load_epoch:
            self.load_epoch = self.args.resume - 1  # 保存的resume名称从1计数
            # 113287是训练样本数量
            self.load_iteration = int(self.args.resume * 113287 / cfg.TRAIN.BATCH_SIZE)

        # 训练优化器
        # load_iteration为scheduler中使用的last_epoch，
        # 用于简单粗略的恢复学习率，只对NoamOpt作用
        # 完整恢复optimizer，还是得保存checkpoint文件
        self.optim = Optimizer(self.model, self.load_iteration)
        # 训练损失计算
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).cuda()
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).cuda()


    # 训练数据集导入
    def setup_dataset(self):
        self.coco_set = datasets.coco_dataset.CocoDataset(
            image_ids_path = cfg.DATA_LOADER.TRAIN_ID,
            input_seq = cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq = cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path = cfg.DATA_LOADER.TRAIN_GV_FEAT,
            att_feats_folder = cfg.DATA_LOADER.TRAIN_ATT_FEATS,
            seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num = cfg.DATA_LOADER.MAX_FEAT
        )

    # DataLoader
    def setup_loader(self, epoch):
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.coco_set)

    # 模型验证
    def eval(self, epoch):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None

        # 验证集上测试结果
        val_res = self.val_evaler(self.model, 'val_' + str(epoch + 1))
        self.logger.info('######## Epoch (VAL)' + str(epoch + 1) + ' ########')
        self.logger.info(str(val_res))

        # 测试集上测试结果
        test_res = self.test_evaler(self.model,'test_' + str(epoch + 1))
        self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
        self.logger.info(str(test_res))

        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
        return val

    def snapshot_path(self, name, epoch):
        # 返回模型路径：experiments/snapshot/{MODELNAME}_{epoch}.pth
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    # 保存模型
    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.model.state_dict(), self.snapshot_path("caption_model", epoch+1))

    def make_kwargs(self, indices, input_seq, target_seq, att_feats, gv_feat):

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.ATT_FEATS: att_feats,
            cfg.PARAM.GLOBAL_FEAT: gv_feat,
        }
        return kwargs

    # 返回scheduled sampling概率
    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            self.model.module.ss_prob = ss_prob

    # 训练数据显示
    def display(self, iteration, data_time, batch_time, losses, loss_info):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}'.format(data_time.avg, batch_time.avg, losses.avg)
        self.logger.info('Iteration ' + str(iteration) + info_str +', lr = ' +  str(self.optim.get_lr()))
        for name in sorted(loss_info):
            self.logger.info('  ' + name + ' = ' + str(loss_info[name]))
        data_time.reset()
        batch_time.reset()
        losses.reset()

    # 模型损失计算过程
    def forward(self, kwargs):
        if self.rl_stage == False:
            # XE训练过程损失计算
            logit = self.model(**kwargs)
            loss, loss_info = self.xe_criterion(logit, kwargs[cfg.PARAM.TARGET_SENT])
        else:
        
            ids = kwargs[cfg.PARAM.INDICES]
            ids = utils.expand_numpy(ids)
    
            kwargs['BEAM_SIZE'] = 5
            kwargs['GREEDY_DECODE'] = False

            # sample sentences 
            seq_sample, logP_sample = self.model.module.decode(**kwargs) # [b*5, 17], [b*5, 17]
            
            rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.detach().cpu().tolist()) # [b*5]

            mask = seq_sample > 0 # [b*5, 17]
            mask = torch.cat([mask.new(mask.shape[0], 1).fill_(1), mask[:, :-1]], dim=1) # consider token <end>
            
            rewards_sample = torch.from_numpy(rewards_sample).cuda().view(-1, 5) # [b, 5]
            reward_baseline = (torch.sum(rewards_sample, dim=1, keepdim=True) - rewards_sample) / (rewards_sample.shape[-1] -1) # [b, 5]

            # ruotian
            # loss = - logP_sample * mask * (rewards_sample - reward_baseline).view(-1, 1) # [b*5, 17]
            # loss = torch.sum(loss) / torch.sum(mask)
           
            loss = - torch.sum(logP_sample * mask, dim=-1) / torch.sum(mask, dim=-1) #  [b*5]
            loss = loss * (rewards_sample - reward_baseline).view(-1)
            loss = loss.mean()

            loss_info = {}
            loss_info['reward_baseline'] = reward_baseline.mean().item()
            return loss, loss_info

        return loss, loss_info

    # 模型训练过程
    def train(self):
        self.model.train()
        # self.optim.zero_grad()

        iteration = self.load_iteration + 1
        # Epoch迭代
        for epoch in range(self.load_epoch + 1, cfg.SOLVER.MAX_EPOCH):
            
            if epoch >= cfg.TRAIN.REINFORCEMENT.START and not self.rl_stage:
                print('------ 开始RL训练 ------')
                self.rl_stage = True

                cfg.SOLVER.BASE_LR = 2e-5 
                cfg.SOLVER.LR_POLICY.TYPE = 'Lambda_rl' 
                
                # cfg.SOLVER.BASE_LR = 5e-6
                # cfg.SOLVER.LR_POLICY.TYPE = 'Fix'
                
                cfg.SOLVER.LR_POLICY.SETP_TYPE = 'Epoch'
                del self.optim
                self.optim = Optimizer(self.model, self.load_iteration)
                cfg.TRAIN.BATCH_SIZE = cfg.TRAIN.RL_BATCH_SIZE

            if 'Lambda' in cfg.SOLVER.LR_POLICY.TYPE:
                self.optim.scheduler_step('Epoch', epoch)            

            print(str(self.optim.get_lr()))
            
            # 设置DataLoader
            self.setup_loader(epoch)

            running_loss = .0
            running_reward_baseline = .0

            # 每一个Epoch内部Iteration迭代
            with tqdm.tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(self.training_loader)) as pbar:
                for _, (indices, input_seq, target_seq, gv_feat, att_feats) in enumerate(self.training_loader):

                    input_seq = input_seq.cuda(non_blocking=True)
                    target_seq = target_seq.cuda(non_blocking=True)
                    att_feats = att_feats.cuda(non_blocking=True)
                    gv_feat = gv_feat.cuda(non_blocking=True)

                    kwargs = self.make_kwargs(indices, input_seq, target_seq, att_feats, gv_feat)
                    
                    loss, loss_info = self.forward(kwargs)
                    loss.backward()
                    # utils.clip_gradient(self.optim.optimizer, self.model, cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                    self.optim.step()
                    
                    # with torch.cuda.amp.autocast():
                    #     loss, loss_info = self.forward(kwargs)   
                    # self.scaler.scale(loss).backward()               
                    # self.scaler.step(self.optim.optimizer)
                    # self.scaler.update()

                    self.optim.zero_grad()

                    if torch.isnan(loss):
                        print('loss is nan !!!')
                        sys.exit()

                    self.optim.scheduler_step('Iter')

                    running_loss += loss.item()

                    if not self.rl_stage:
                        pbar.set_postfix(loss='%.4f' % (running_loss / (_ + 1)))
                    else:
                        running_reward_baseline += loss_info['reward_baseline']
                        pbar.set_postfix({'loss/r_b': '%.4f/%.4f' % (running_loss / (_ + 1), running_reward_baseline / (_ + 1))})
                    pbar.update()
                    iteration += 1

                    if self.distributed:
                        dist.barrier()
            

            # 每一个Epoch结束保存模型
            self.save_model(epoch)
            val = self.eval(epoch)
            
            if 'Lambda' not in cfg.SOLVER.LR_POLICY.TYPE:
                self.optim.scheduler_step('Epoch', epoch)
            
            if self.distributed:
                dist.barrier()

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--load_epoch", action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    trainer = Trainer(args)
    trainer.train()
# """

"""
train_coco_set = datasets.coco_dataset.CocoDataset(
    image_ids_path = './mscoco/txt/coco_train_image_id.txt',
    input_seq = './mscoco/sent/coco_train_input.pkl',
    target_seq = './mscoco/sent/coco_train_target.pkl',
    gv_feat_path = '',
    att_feats_folder = './mscoco/feature/grid_X_101', 
    seq_per_img = 5,
    max_feat_num = -1
)

training_loader = datasets.data_loader.load_train(
    False,
    1, 
    train_coco_set
)

t1 = time.time()
for _, (indices, input_seq, target_seq, gv_feat, att_feats, att_mask) in enumerate(training_loader):
    t2 = time.time()
    print(_, input_seq.shape, target_seq.shape, gv_feat.shape, att_feats.shape, att_mask.shape, t2-t1)
    t1 = t2

"""
