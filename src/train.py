import argparse
import numpy as np
from model import HJ_Model
import dataloader
import torch
torch.set_printoptions(profile='full')

def isRight(pred, label):
    return 1 if torch.argmax(label, dim=1) == torch.argmax(pred, dim=1) else 0

def printInfo(args, epoch, trainLoss, accRate):
    print('epoch:', epoch, '/', args.epoch, end=' ')
    print('train loss:', round(trainLoss, 4), end=' ')
    print('acc rate:', round(accRate * 100, 1), '%')

def train(args):
    dataset = RTLDesignDataset(args.dataset, args.ndim)
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    if args.do_pretrain:
        pass
    elif args.finetune_from != None:
        pass
    else:
        model = HJ_Model(args.ndim, 2)
        lossFunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)

    d_trainRecord = {'meanTrainLoss':[], 'trainAccRate':[]}
    for epoch in range(0, args.epoch):
        l_trainLoss = []
        l_isRight = []
        model.train()
        for i, (sample, label) in enumerate(dataset):
            # print(sample.edge_index.T)
            out = model(sample)
            label = torch.reshape(label, (1, 2))
            loss = lossFunc(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l_trainLoss.append(float(loss))
            l_isRight.append(isRight(out, label))
        meanTrainLoss = np.mean(l_trainLoss)
        accRate = np.mean(l_isRight)
        printInfo(args, epoch, meanTrainLoss, accRate)
        d_trainRecord['meanTrainLoss'].append(meanTrainLoss)
        d_trainRecord['trainAccRate'].append(accRate)

def parseArgs():
    parser = argparse.ArgumentParser(description='GPT4HJ training')
    parser.add_argument('--dataset', type=str, default='./datasets',
                        help='directory of dataset')
    parser.add_argument('--epoch', type=int, default=200,
                        help='epoch number')
    parser.add_argument('--lr', type=int, default=1e-5,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--do_pretrain', action='store_true', default=False,
                        help='do pretraining')
    parser.add_argument('--finetune_from', type=str, default=None,
                        help='finetune from pretrained model')
    parser.add_argument('--ndim', type=int, default=256,
                        help='dimension of node feature')
    args = parser.parse_args()
    assert(not (args.do_pretrain and args.finetune_from != None))
    return args

if __name__ == '__main__':
    args = parseArgs()
    train(args)
