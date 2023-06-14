import argparse
import numpy as np
from model import HJ_Model
import dataloader
import torch
torch.set_printoptions(profile='full')
LABEL={'TjFree': [0., 1.], 
       'TjIn': [1., 0.]}

def isRight(pred, label):
    return 1 if torch.argmax(label, dim=1) == torch.argmax(pred, dim=1) else 0

def printInfo(args, epoch, trainLoss, trainAccRate, devLoss, devAccRate):
    print('epoch:', epoch, '/', args.epoch, end=' ')
    print('train loss:', round(trainLoss, 4), end=' ')
    print('train acc rate:', round(trainAccRate * 100, 1), '%')
    print('train loss:', round(devLoss, 4), end=' ')
    print('dev acc rate:', round(devAccRate * 100, 1), '%')

def trainprocess(dataloader: dataloader.DataLoader,
                 model: torch.nn.Module,
                 backward: bool,
                 lossFunc,
                 optimizer):
    losslist = []
    rightlist = []
    for ast in dataloader:
        out = model(ast.toPyG())
        label = torch.tensor(LABEL[ast.label])
        label = torch.reshape(label, (1, 2))
        loss = lossFunc(out, label)
        optimizer.zero_grad()
        if backward:
            loss.backward()
            optimizer.step()
        losslist.append(float(loss))
        rightlist.append(isRight(out, label))
    info = {'losslist':losslist, 'rightlist':rightlist}
    return info

def train(args):
    recipeloader = dataloader.RecipeLoader(args.dataset)
    astloader = dataloader.ASTLoader(recipeloader)
    (trainloader, devloader) = dataloader.divide(astloader, args.devname)
    model = HJ_Model(5, 2)
    lossFunc = torch.nn.CrossEntropyLoss()
    # lossFunc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
    # start training
    d_trainRecord = {'meanTrainLoss':[], 'trainAccRate':[]}
    for epoch in range(0, args.epoch):
        # train
        model.train()
        info = trainprocess(trainloader, model, True, lossFunc, optimizer)
        meanTrainLoss = np.mean(info['losslist'])
        trainAccRate = np.mean(info['rightlist'])
        # dev
        with torch.no_grad():
            model.eval()
            info = trainprocess(devloader, model, False, lossFunc, optimizer)
        meanDevLoss = np.mean(info['losslist'])
        devAccRate = np.mean(info['rightlist'])
        printInfo(args, epoch, meanTrainLoss, trainAccRate, meanDevLoss, devAccRate)
        d_trainRecord['meanTrainLoss'].append(meanTrainLoss)
        d_trainRecord['trainAccRate'].append(trainAccRate)

def parseArgs():
    parser = argparse.ArgumentParser(description='HTrans')
    parser.add_argument('--dataset', type=str, default='./datasets',
                        help='directory of dataset')
    parser.add_argument('--epoch', type=int, default=200,
                        help='epoch number')
    parser.add_argument('--lr', type=int, default=1e-5,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--devname', type=str, default='RS232',
                        help='devset name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    train(args)
