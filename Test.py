from Model.Network import myNetwork
from Option import args
from Utils import *
import cv2
from Model.LF_Operator import sai2mpi,mpi2sai
import torch.nn as nn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model = myNetwork(dim=args.channel, num_heads=args.num_heads)
load_checkpoint(model, os.path.join(args.log_path, 'model_best.pth'))
model = nn.DataParallel(model)
model.cuda()

model.eval()

for scene in os.listdir(args.test_path):
    scene_path = os.path.join(args.test_path, scene)
    for num in range(3):
        input_path = os.path.join(scene_path, 'rainy-' + str(num+1))
        input_img = torch.zeros([1, 3, 1600, 2000], dtype=torch.float).cuda()
        for i in range(5):
            for j in range(5):
                img_path = os.path.join(input_path, f"{i+1}_{j+1}.jpg")
                img_tmp = cv2.imread(img_path)
                img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
                img_tmp = torch.from_numpy(img_tmp).permute(2, 0, 1)/255.0
                input_img[:, :, i*320:(i+1)*320, j*400:(j+1)*400] = img_tmp.cuda()
        input_img = sai2mpi(input_img)
        input_img = input_img.unfold(2, 400, 400).unfold(3, 400, 400)
        input_img = input_img.permute(0, 2, 3, 1, 4, 5).contiguous()
        input_img = input_img.view(-1, 3, 400, 400)
        input_img = mpi2sai(input_img)
        with torch.no_grad():
            for n in range(20):
                img = input_img[n, :, :, :]
                img = img.unsqueeze(0)
                out = model(img)
                input_img[n, :, :, :] = out
        input_img = sai2mpi(input_img)
        input_img = input_img.view(1, 4, 5, 3, 400, 400)
        input_img = input_img.permute(0, 3, 1, 4, 2, 5).contiguous()
        input_img = input_img.view(1, 3, 4 * 400, 5 * 400)
        input_img = mpi2sai(input_img)
        input_img = input_img.clamp(0, 1)
        input_img = (input_img[0].cpu().numpy() * 255).astype('uint8')
        input_img = cv2.cvtColor(input_img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        img_name = os.path.join(args.result_path, scene)
        os.makedirs(img_name, exist_ok=True)
        img_name = os.path.join(img_name, str(num+1) + '.jpg')
        cv2.imwrite(img_name, input_img)
