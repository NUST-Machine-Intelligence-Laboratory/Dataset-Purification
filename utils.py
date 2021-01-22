from torch.utils.data import DataLoader
from dataset.Imagefolder_modified import Imagefolder_modified
from resnet import *
from PIL import ImageFile # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# plt.rcParams['axes.linewidth'] = 2

# 计算所有样本的与类别中心的cos距离
def cos_compute(data_dir, net=None, model_dir='model/resnet50sub_best.pth',n_class=200):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=448),
        torchvision.transforms.CenterCrop(size=448),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_data = Imagefolder_modified(os.path.join(data_dir, 'train'), transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=8,
                              shuffle=False, num_workers=4, pin_memory=True)

    if net == 'resnet18_sub':
        NET = ResNet18_subcenter
    elif net == 'resnet18_ss':
        NET = ResNet18_ss
    elif net == 'resnet50_sub':
        NET = ResNet50_subcenter
    else:
        raise AssertionError('Not implemented yet')

    model = NET(pretrained=False,n_classes=n_class)
    model = torch.nn.DataParallel(model).cuda()

    chkpt = torch.load(model_dir)
    model.load_state_dict(chkpt)
    model.train(False)

    res = torch.zeros((n_class, 3))
    record = torch.zeros((len(train_data), 2))

    for X, y, id, path in train_loader:
        # Data
        X = X.cuda()
        y = y.cuda()

        cos_angle, feat = model(X)
        pre = torch.argmax(cos_angle, dim=1)
        center = torch.argmax(feat, dim=2)

        for i in range(y.size(0)):
            res[pre[i], center[i, pre[i]]] += 1
            record[id[i], 0] = pre[i]  # 记录标签
            record[id[i], 1] = center[i, pre[i]]  # 记录中心

    dor = torch.argmax(res, dim=1)

    data=[]
    for X, y, id, path in train_loader:
        # Data
        X = X.cuda()
        y = y
        _, feat = model(X)
        for i in range(feat.shape[0]):
            cos = feat[i,y[i],dor[y[i]]]
            cos = cos.detach().cpu()
            if record[id[i], 1] == dor[int(record[id[i], 0])]:
                d = 1
            else:
                d = 0
            tmp=[path[i], int(y[i]), float(cos), d]  # 路径，标签，cos
            data.append(tmp)

    f = open('pkls/data.pkl', 'wb')
    pickle.dump(data, f)
    f.close()

# 画直方图并生成 noise_list.txt 名单
def noise_identify(thr=60):
    f = open('pkls/data.pkl', 'rb')
    data = pickle.load(f)
    f.close()

    data.sort(key=lambda x: x[2])

    dor_angles = [x[2] for x in data if x[3] == 1]
    sub_angles = [x[2] for x in data if x[3] == 0]
    angles = [x[2] for x in data]

    sub_angles = np.arccos(np.array(sub_angles))
    sub_angles = np.degrees(np.sort(sub_angles))
    dor_angles = np.arccos(np.array(dor_angles))
    dor_angles = np.degrees(np.sort(dor_angles))
    angles = np.arccos(np.array(angles))
    angles = np.degrees(np.sort(angles))
    print(min(sub_angles))

    noise = np.where(angles > thr)
    print(noise[0].shape[0])

    sub = pd.DataFrame(sub_angles)
    t = sub.hist(bins=100, density = False, alpha=0.5, edgecolor='blue')
    dor = pd.DataFrame(dor_angles)
    dor.hist(bins=100, density = False, ax=t, alpha=0.5, edgecolor='orange')

    plt.grid(axis="y", linestyle='None')
    plt.grid(axis="x", linestyle='None')
    plt.ylim((0,1200))
    plt.title('')
    plt.xlabel('Angle between Samples and Dominant Sub-center.')
    plt.ylabel('Image Numbers')
    plt.legend(['Non-dominant Sub-center Samples', 'Dominant Sub-center Samples'], loc='upper right', edgecolor='black', fontsize=11)
    plt.plot((thr, thr), (0, 1200), color='r', linestyle='--')

    plt.savefig('cub_resnet50.png', dpi=2500, bbox_inches = 'tight')
    plt.show()

    noi=data[:noise[0].shape[0]]
    noise_list=[]
    f = open('dataset/noise_list_thr{}.pkl'.format(thr), 'wb')
    for i in range(len(noi)):
        tmp = [noi[i][0], int(noi[i][1]), float(noi[i][2])]
        noise_list.append(tmp)
    pickle.dump(noise_list, f)
    f.close()

    clean=data[noise[0].shape[0]:]
    clean_list=[]
    f = open('dataset/clean_list_thr{}.pkl'.format(thr), 'wb')
    for i in range(len(clean)):
        tmp = [clean[i][0], int(clean[i][1]), float(clean[i][2])]
        clean_list.append(tmp)
    pickle.dump(clean_list, f)
    f.close()


def gen_ind_list(pkl='pkls/relabel.pkl', number=500, record = False):
    f = open(pkl, 'rb')
    data = pickle.load(f)
    f.close()

    data.sort(key=lambda x: x[1], reverse=True)
    score = [x[1] for x in data]
    score = np.array(score)
    dor = pd.DataFrame(score)
    dor.hist(bins=100, density=False, alpha=0.5)
    plt.grid(axis="y", linestyle='--')

    plt.title('')
    plt.show()

    relabel_list=[]
    f = open('dataset/relabel_list_thr60_{}.pkl'.format(number), 'wb')
    images=[]

    for i in range(number):
        # path, pseudo label
        tmp = [data[i][0], int(data[i][2])]
        relabel_list.append(tmp)

        if record:
            image = Image.open(data[i][0])
            image = image.convert('RGB')
            image = image.resize((224, 224), Image.ANTIALIAS)
            images.append(image)

    pickle.dump(relabel_list, f)
    f.close()

    if record:
        UNIT_SIZE = 224  #
        TARGET_WIDTH = UNIT_SIZE * 6  # 拼接完后的横向长度
        TARGET_Height = UNIT_SIZE * 8
        target = Image.new('RGB', (TARGET_WIDTH, TARGET_Height))
        count = 0
        for t in range(int(len(images) / 48)):
            for i in range(6):
                for j in range(8):
                    target.paste(images[count], (UNIT_SIZE * i, UNIT_SIZE * j))
                    count += 1
            quality_value = 100  # quality来指定生成图片的质量，范围是0～100
            target.save('image/' + '{}.jpg'.format(t), quality=quality_value)


if __name__ == '__main__':
    # cos_compute()
    noise_identify()
    # gen_ind_list()
