import os
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path
import Utils
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from Option import args
from Model.Network import myNetwork
from Loss import PSNRLoss
from Dataset import *


dir_checkpoint = Path('./checkpoints/')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if args.preprocess:
    crop_data(args.train_path, 80, [80,80], args.A)
    crop_data(args.val_path, 80, [80, 80], args.A)

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1

result_dir = os.path.join(args.log_path, 'results')
model_dir = os.path.join(args.log_path, 'models')
Utils.mkdir(result_dir)
Utils.mkdir(model_dir)

######### Model ###########
model_restoration = myNetwork(dim=args.channel, num_heads=args.num_heads)
print("Total number of param  is ", sum(x.numel() for x in model_restoration.parameters()))
model_restoration.cuda()

######### Scheduler ###########
new_lr = args.lr
optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch - warmup_epochs, eta_min=args.lr_min)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if args.Resume:
    path_chk_rest = os.path.join(dir_checkpoint, 'model_best.pth')
    Utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = Utils.load_start_epoch(path_chk_rest) + 1
    Utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
      scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

######### Loss ###########
criterion = PSNRLoss().cuda()

######### DataLoaders ###########
train_dataset = Dataset(mode='train', data_path=args.train_path)
train_loader = DataLoader(dataset=train_dataset, num_workers=1, batch_size=args.batch_size, shuffle=True)

val_dataset = Dataset(mode='val', data_path=args.val_path)
val_loader = DataLoader(dataset=val_dataset, num_workers=1, batch_size=args.batch_size, shuffle=False)

best_psnr = 0
best_epoch = 0
global_step = 0

for epoch in range(start_epoch, args.epoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    psnr_train_rgb = []
    psnr_tr = 0
    psnr_tr1 = 0
    ssim_tr1 = 0

    for i, data in enumerate(tqdm(train_loader), 0):
        model_restoration.train()
        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        rainy = data[0].cuda()
        target = data[1].cuda()

        restored = model_restoration(rainy)

        # Compute loss at each stage
        loss = criterion(target, restored)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        global_step = global_step + 1

    psnr_te = 0
    psnr_te_1 = 0
    ssim_te_1 = 0
    #### Evaluation ####
    if epoch % args.val_after_every == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            rainy = data_val[0].cuda()
            target = data_val[1].cuda()

            with torch.no_grad():
                restored = model_restoration(rainy)

            restored = torch.clamp(restored, 0., 1.)
            tssss = Utils.batch_PSNR(restored, target, 1.)
            print('PSNR: ', tssss)
            psnr_te = psnr_te + tssss
            psnr_val_rgb.append(tssss)

        psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
        print("te", psnr_te)
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, str(dir_checkpoint / "model_best.pth"))

        print(
            "[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
