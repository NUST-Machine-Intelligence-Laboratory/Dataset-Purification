#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import torch.nn.functional as F
from dataset.dataset_from_list import dataset_from_list

from PIL import ImageFile # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import *
import time
import pickle

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def ind_score(pred, n=4):
    smax = F.softmax(pred, 1)
    res = torch.zeros(int(smax.size(0) / n))
    for i in range(res.size(0)):
        for j in range(n):
            std = torch.std(smax[i + res.size(0) * j])
            res[i] += std
    return res/n

class Manager_AM(object):
    def __init__(self, options):
        """
        Prepare the network, criterion, Optimizer and data
        Arguments:
            options [dict]  Hyperparameter
            path    [dict]  path of the dataset and model
        """
        print('------------------------------------------------------------------------------')
        print('Preparing the network and data ... ')
        self._options = options
        self._path = options['path']
        os.popen('mkdir -p ' + self._path)
        self._data_base = options['data_base']
        self._class = options['n_classes']
        self._data_list = options['data_list']
        self._step = options['step']
        self._smooth = options['smooth']

        print('Basic information: ', 'data:', self._data_base, '    lr:', self._options['base_lr'], ' w_decay:', self._options['weight_decay'])
        print('Parameter information: ', 'step:', self._step, ' smooth:', self._smooth)

        # Network
        if options['net'] == 'resnet18_sub':
            NET = ResNet18_subcenter
        elif options['net'] == 'resnet18_ss':
            NET = ResNet18_ss
        elif options['net'] == 'resnet50_sub':
            NET = ResNet50_subcenter
        else:
            raise AssertionError('Not implemented yet')

        if self._step !=2:
            net = NET(n_classes=options['n_classes'], pretrained=True)
        else:
            net = NET(n_classes=options['n_classes'], pretrained=False)

        if torch.cuda.device_count() >= 1:
            self._net = torch.nn.DataParallel(net).cuda()
            print('cuda device : ', torch.cuda.device_count())
        else:
            raise EnvironmentError('This is designed to run on GPU but no GPU is found')
        # Criterion
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Optimizer

        params_to_optimize = self._net.parameters()

        self._optimizer = torch.optim.SGD(params_to_optimize, lr=self._options['base_lr'],
                                          momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self._options['epochs'])

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        # Load data
        if self._data_list == None:
            train_data = Imagefolder_modified(os.path.join(self._data_base, 'train'), transform=train_transform)
        else:
            train_data = dataset_from_list(self._data_list, transform=train_transform)
        test_data = torchvision.datasets.ImageFolder(os.path.join(self._data_base, 'val'), transform=test_transform)
        self._train_loader = DataLoader(train_data, batch_size=self._options['batch_size'],
                                        shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = DataLoader(test_data, batch_size=16,
                                       shuffle=False, num_workers=4, pin_memory=True)


    def _label_smoothing_cross_entropy(self,logit, label, epsilon=0.1, reduction='mean'):
        N = label.size(0)
        C = logit.size(1)
        smoothed_label = torch.full(size=(N, C), fill_value=epsilon / (C - 1))
        smoothed_label.scatter_(dim=1, index=torch.unsqueeze(label, dim=1).cpu(), value=1 - epsilon)

        if logit.is_cuda:
            smoothed_label = smoothed_label.cuda()

        log_logit = F.log_softmax(logit, dim=1)
        losses = -torch.sum(log_logit * smoothed_label, dim=1)  # (N)
        if reduction == 'none':
            return losses
        elif reduction == 'mean':
            return torch.sum(losses) / N
        elif reduction == 'sum':
            return torch.sum(losses)
        else:
            raise AssertionError('reduction has to be none, mean or sum')

    def train(self):
        """
        Train the network
        """
        print('Training ... ')
        best_accuracy = 0.0
        best_epoch = None
        print('Epoch\tTrain Loss\tTrain Accuracy\tTest Accuracy\tEpoch Runtime')
        s = 30
        for t in range(self._options['epochs']):
            epoch_start = time.time()
            epoch_loss = []
            record=[]
            num_correct = 0
            num_total = 0
            num_train_total = 0
            # self._classweigt_tmp = torch.zeros(self._class).cuda()
            for X, y, id, path in self._train_loader:
                # Enable training mode
                self._net.train(True)
                # Data
                X = X.cuda()
                y = y.cuda()
                # Forward pass
                cos_angle,_ = self._net(X)  # score is in shape (N, 200)
                # pytorch only takes label as [0, num_classes) to calculate loss
                cos_angle = torch.clamp(cos_angle, min=-1, max=1)
                weighted_cos_angle = s * cos_angle

                if self._smooth <=0:
                    loss = self._criterion(weighted_cos_angle, y)
                else:
                    loss = self._label_smoothing_cross_entropy(weighted_cos_angle, y, epsilon=self._smooth)

                num_train = y.size(0)

                epoch_loss.append(loss.item())
                # Prediction
                closest_dis, prediction = torch.max(cos_angle.data, 1)

                # prediction is the index location of the maximum value found,
                num_total += y.size(0)  # y.size(0) is the batch size
                num_correct += torch.sum(prediction == y.data).item()
                num_train_total += num_train
                # Clear the existing gradients
                self._optimizer.zero_grad()
                # Backward
                loss.backward()
                self._optimizer.step()
            # Record the train accuracy of each epoch
            train_accuracy = 100 * num_correct / num_total
            test_accuracy = self.test(self._test_loader)
            self._scheduler.step()  # the scheduler adjust lr based on test_accuracy

            epoch_end = time.time()

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = t + 1  # t starts from 0
                print('*', end='')
                # Save mode
                torch.save(self._net.state_dict(), os.path.join(self._path, self._options['net'] + 'best.pth'))
            # if t % 10 == 0:
            #     torch.save(self._net.state_dict(), os.path.join(self._path, options['net'] + '_{}.pth'.format(t)))
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f\t\t%4.2f' % (t + 1, sum(epoch_loss) / len(epoch_loss),
                                                            train_accuracy, test_accuracy,
                                                            epoch_end - epoch_start, num_train_total))

        print('-----------------------------------------------------------------')
        print('Best at epoch %d, test accuracy %f' % (best_epoch, best_accuracy))
        print('-----------------------------------------------------------------')

    def train_self_supervised(self):
        """
        Train the network
        """
        print('Step3 self-supervised training ... ')
        best_accuracy = 0.0
        best_epoch = None
        print('Epoch\tTrain Loss\tTrain Accuracy grey\tTrain Accuracy rot\tTest Accuracy grey\tTest Accuracy Rot\tEpoch Runtime')
        for t in range(self._options['epochs']):
            epoch_start = time.time()
            epoch_loss = []
            num_correct_rot = 0
            num_total_rot = 0
            data = []
            for X, y, _, path in self._train_loader:
                # Enable training mode
                self._net.train(True)
                # Data
                X = X.cuda()
                loss = torch.FloatTensor([0]).cuda()

                # self-supervised task rot
                y_prime = torch.cat((torch.zeros(X.size(0)), torch.ones(X.size(0)),
                                     2 * torch.ones(X.size(0)), 3 * torch.ones(X.size(0))), 0).long()
                X_rot = torch.cat((X, torch.rot90(X, 1, dims=[2, 3]),
                               torch.rot90(X, 2, dims=[2, 3]), torch.rot90(X, 3, dims=[2, 3])), 0)
                X_rot, y_prime = X_rot.cuda(), y_prime.cuda()

                _, rot_pred = self._net(X_rot)

                if self._smooth <= 0:
                    loss += self._criterion(rot_pred, y_prime)
                else:
                    loss += self._label_smoothing_cross_entropy(rot_pred, y_prime, epsilon=self._smooth)

                # Prediction
                _, prediction = torch.max(rot_pred.data, 1)
                num_total_rot += y_prime.size(0)  # y.size(0) is the batch size
                num_correct_rot += torch.sum(prediction == y_prime.data).item()
                epoch_loss.append(loss.item())

                # Clear the existing gradients
                self._optimizer.zero_grad()
                # Backward
                loss.backward()
                self._optimizer.step()
            # Record the train accuracy of each epoch
            train_accuracy_rot = 100 * num_correct_rot / num_total_rot
            test_accuracy_rot = self.test(self._test_loader,ss='rot')

            self._scheduler.step()
            epoch_end = time.time()

            if test_accuracy_rot > best_accuracy:
                best_accuracy = test_accuracy_rot
                best_epoch = t + 1  # t starts from 0
                print('*', end='')
                # Save mode
                torch.save(self._net.state_dict(), os.path.join(self._path, self._options['net'] + 'best.pth'))
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f\t\t%4.2f' % (t + 1, sum(epoch_loss) / len(epoch_loss),
                                                                        train_accuracy_rot, test_accuracy_rot,
                                                                        epoch_end - epoch_start, num_total_rot))

        print('-----------------------------------------------------------------')
        print('Best at epoch %d, test accuracy %f' % (best_epoch, best_accuracy))
        print('-----------------------------------------------------------------')

    def ood_detection(self, noise_list='dataset/noise_list_thr61.pkl', ss_path='model/step3/resnet18_ssbest.pth', model_path='model/step3/resnet50_subbest.pth'):
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        noise_data = dataset_from_list(noise_list, transform=test_transform)
        noise_loader=DataLoader(noise_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        self._net.load_state_dict(torch.load(ss_path))
        self._net.train(False)  # set the mode to evaluation phase

        net2=ResNet50_subcenter(n_classes=self._class, pretrained=False)
        net2=torch.nn.DataParallel(net2).cuda()
        net2.load_state_dict(torch.load(model_path))
        net2.train(False)  # set the mode to evaluation phase
        record = []

        with torch.no_grad():
            for X, y, ids, path in noise_loader:
                # Data
                X = X.cuda()
                score, _ = net2(X)
                # softmax, _ = torch.max(F.softmax(score, 1),1)
                _, label = torch.max(score,1)

                # rot
                y_prime = torch.cat((torch.zeros(X.size(0)), torch.ones(X.size(0)),
                                     2 * torch.ones(X.size(0)), 3 * torch.ones(X.size(0))), 0).long()
                X_rot = torch.cat((X, torch.rot90(X, 1, dims=[2, 3]),
                               torch.rot90(X, 2, dims=[2, 3]), torch.rot90(X, 3, dims=[2, 3])), 0)
                X_rot, y_prime = X_rot.cuda(), y_prime.cuda()

                _, rot_pred = self._net(X_rot)  # score is in shape (N, 200)

                rot_score = ind_score(rot_pred.clone().detach().cpu())
                for i in range(y.size(0)):
                    temp = []
                    temp.append(path[i])
                    temp.append(float(rot_score[i].clone().detach()))
                    temp.append(int(label[i].clone().detach()))
                    record.append(temp)

        # record.sort(key=lambda x: x[0])  # ascending order

        f = open('pkls/relabel.pkl', 'wb')
        pickle.dump(record, f)
        f.close()
        return

    def test(self, dataloader, ss = None):
        """
        Compute the test accuracy

        Argument:
            dataloader  Test dataloader
        Return:
            Test accuracy in percentage
        """
        self._net.train(False) # set the mode to evaluation phase
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for X, y in dataloader:
                # Data
                X = X.cuda()
                y = y.cuda()
                if ss == 'rot':
                    y_prime = torch.cat((torch.zeros(X.size(0)), torch.ones(X.size(0)),
                                          2 * torch.ones(X.size(0)), 3 * torch.ones(X.size(0))), 0).long()
                    X = torch.cat((X, torch.rot90(X, 1, dims=[2, 3]),
                                         torch.rot90(X, 2, dims=[2, 3]), torch.rot90(X, 3, dims=[2, 3])), 0)
                    X, y_prime = X.cuda(), y_prime.cuda()

                    _, rot_pred = self._net(X)  # score is in shape (N, 200)
                    # Prediction
                    _, prediction = torch.max(rot_pred.data, 1)
                    num_total += y_prime.size(0)  # y.size(0) is the batch size
                    num_correct += torch.sum(prediction == y_prime.data).item()
                # Prediction
                else:
                    score,_ = self._net(X)
                    _, prediction = torch.max(score, 1)
                    num_total += y.size(0)
                    num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # set the mode to training phase
        return 100 * num_correct / num_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
    parser.add_argument('--net', dest='net', type=str, default='resnet18',
                        help='supported options: resnet18, resnet50, bcnn')
    parser.add_argument('--n_classes', dest='n_classes', type=int, default=200,
                        help='number of classes')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-2)
    parser.add_argument('--w_decay', dest='weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--path', dest='path', type=str, default='model')
    parser.add_argument('--data_base', dest='data_base', type=str, default='/home/zcy/data/fg-web-data/web-bird')
    parser.add_argument('--dl', nargs='+', dest='data_list', type=str, default=None)
    parser.add_argument('--step', dest='step', type=int, default=1)
    parser.add_argument('--smooth', dest='smooth',  type=float, default=0)

    args = parser.parse_args()

    model = args.path

    print(os.path.join(os.popen('pwd').read().strip(), model))

    if not os.path.isdir(os.path.join(os.popen('pwd').read().strip(), model)):
        print('>>>>>> Creating directory \'model\' ... ')
        os.mkdir(os.path.join(os.popen('pwd').read().strip(), model))

    path = os.path.join(os.popen('pwd').read().strip(), model)

    options = {
            'base_lr': args.base_lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'path': path,
            'data_base': args.data_base,
            'net': args.net,
            'n_classes': args.n_classes,
            'data_list': args.data_list,
            'step': args.step,
            'smooth':args.smooth
        }
    if args.step == 1 or (args.step == 3 and args.net == 'resnet50_sub') or args.step == 4:
        manager = Manager_AM(options)
        manager.train()
    elif args.step == 2 :
        cos_compute(net=args.net, data_dir=args.data_base, model_dir=path + '/resnet50_subbest.pth')
        noise_identify(thr=61)
    elif args.step == 3:
        manager = Manager_AM(options)
        manager.train_self_supervised()
        manager.ood_detection()
        gen_ind_list()