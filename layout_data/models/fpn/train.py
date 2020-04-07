# encoding: utf-8
"""
@author:  zhouzikun
@contact:  zikun_zhou@shannonai.com

@version:  1.1
@file:  shannonocr - model
@time:  2019/10/7 0:07

"""
import os

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from fpn.model import fpn                                     #从fpn文件中model导入fpn类
from utils.segmentation_evaluation import validate
from utils import project_path
from utils.mat2pic import GeneralDataset, TestDataset, trans_separate
from utils.model_init import weights_init, weights_init_without_kaiming
from utils.focalloss import FocalLoss
from utils.scheduler import WarmupMultiStepLR

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        print("Use CPU")
        device = torch.device('cpu')
    else:
        print("Use GPU:", os.environ['CUDA_VISIBLE_DEVICES'])

    batch_size = 64
    max_epochs = 200
    model_save_interval = 2
    model_valid_interval = 1
    LEARNING_RATE = 1e-3
    LOAD_PRETRAIN = False

    model = fpn().to(device) #导入模型

    model_path = os.path.join(project_path, 'data', 'fpn.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    log_dir = os.path.join(project_path, 'log', 'fpn')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    dataset = GeneralDataset(trans_separate, resize_shape=(200, 200))    #生成数据集，trans_separate是什么参数
    dataset_test = TestDataset(trans_separate, resize_shape=(200, 200))

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_test, batch_size = 16, shuffle=False, drop_last=True)

    print("model path:", model_path)

    if LOAD_PRETRAIN and os.path.exists(model_path):
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("model initiated with", model_path)
    else:
        model.backbone.apply(weights_init)# TODO@LYC: Init Header
        # model.head.apply(weights_init_without_kaiming) # not very effective
        print("model initiated without pretrain")
    for p in model.parameters():
        p.requires_grad = True

    print("\tLearning Rate:", LEARNING_RATE)
    print("\tBatch Size:", batch_size)
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = WarmupMultiStepLR(optimizer, milestones=[], warmup_iters=len(train_loader))

    criterion = FocalLoss()

    USE_NORMAL_LOSS = True # 截断大loss

    for epoch in range(max_epochs):
        if USE_NORMAL_LOSS:
            normal_loss = -1
        for it, images in enumerate(train_loader):
            layout_image = images[0].to(device)
            heat_image = images[1].to(device)
            m = model(heat_image)  #这里是热场到布局图的映射

            # loss a= F.binary_cross_entropy(m, layout_image, reduction="mean")
            loss = criterion(m, layout_image)

            if USE_NORMAL_LOSS and epoch > 0:
                if normal_loss == -1:
                    normal_loss = loss.item()
                if loss.item() > normal_loss * 10:
                    writer.add_scalar('LOSS', -1, epoch * len(train_loader) + it)
                    continue
                normal_loss = normal_loss * 0.9 + loss.item() * 0.1

            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("\tEpoch[{}/{}] Iters[{}/{}] Loss: {:.3f}".format(
                epoch + 1, max_epochs,
                it, len(train_loader),
                loss.item()*1e5),
            )
            writer.add_scalar('LOSS', loss.item(), epoch * len(train_loader) + it)

        if epoch % model_valid_interval == 0:
            model.eval()
            print("Model Validation")
            with torch.no_grad():
                mae, accuracy = validate(model, valid_loader, max_iter=-1, flag_detail=False)
                print("1-MAE:", round(mae*100,4) , 'Accuracy:',round(accuracy * 100, 4))
            model.train()

            writer.add_scalar('1-MAE', mae, epoch)
            writer.add_scalar('ACC', accuracy, epoch)

        if epoch % model_save_interval == 0:
            torch.save(model.state_dict(), model_path+"."+str(epoch))
            print("Model Saved:", model_path+"."+str(epoch))


