import argparse
import numpy as np
from model import HJ_Model, HJ_LocModel
import dataloader
import torch
torch.set_printoptions(profile='full')
LABEL={'TjFree': [1., 0.], 
       'TjIn': [0., 1.]}

def isRight(pred_k_nearest: list, labels: list) -> int:
    """
    Input: prediction, label
    Output: 0 / 1
    """
    for pred in pred_k_nearest:
        for label in labels:
            if pred[1]['token'] == label:
                return 1
    print([node[1]['token'] for node in pred_k_nearest])
    print(labels)
    return 0

def cal_metric(tp, fp, fn):
    precision = tp / (tp + fp + 1e-20)
    recall = tp / (tp + fn + 1e-20)
    F1score = 2 * precision * recall / (precision + recall + 1e-20)
    return [precision, recall, F1score]

def printInfo(args, epoch, trainLoss, trainAccRate, devLoss, devAccRate, errlist):
    print('epoch:', epoch, '/', args.epoch, end=' ')
    print('train loss:', round(trainLoss, 4), end=' ')
    print('train acc:', round(trainAccRate * 100, 1), '%')
    print('dev loss:', round(devLoss, 4), end=' ')
    print('dev acc:', round(devAccRate * 100, 1), '%')
    print('error:', errlist)

def trainprocess(dataloader: dataloader.DataLoader,
                 model: torch.nn.Module,
                 backward: bool,
                 lossFunc,
                 optimizer):
    losslist = []
    rightlist = []
    errlist = []
    for ast in dataloader:
        out = model(ast.toPyG())
        label = ast.TjLoc_Feature()
        loss = lossFunc(out, label)
        optimizer.zero_grad()
        if backward:
            loss.backward()
            optimizer.step()
        losslist.append(float(loss))
        # token = ast.find_k_nearest(label, lossFunc, 1)[0][1]['token']
        # if token != ast.TjLoc:
        #     print(token, ast.TjLoc)
        #     ast.ast.show()
        #     exit()
        right = isRight(ast.find_k_nearest(out, lossFunc, 5), ast.TjLoc)
        # if right == 0:
        #     print(ast.name)
        #     ast.ast.show()
        #     exit()
        rightlist.append(right)
        if right == 0:
            errlist.append(ast.name + ast.label) 
    info = {'losslist':losslist, 'rightlist':rightlist, 'errlist': errlist}
    return info

def train(args):
    recipeloader = dataloader.RecipeLoader(args.dataset, True)
    astloader = dataloader.ASTLoader(recipeloader, True)
    (trainloader, devloader) = dataloader.divide(astloader, args.devname)
    print('trainset len:', len(trainloader), 'devset len:', len(devloader))
    model = HJ_LocModel(5, 2)
    # lossFunc = torch.nn.CrossEntropyLoss()
    lossFunc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
    # start training
    record = {'meanTrainLoss':[], 'acc':[]}
    for epoch in range(0, args.epoch):
        # train
        model.train()
        info = trainprocess(trainloader, model, True, lossFunc, optimizer)
        meanTrainLoss = np.mean(info['losslist'])
        trainAcc = np.mean(info['rightlist'])
        # dev
        with torch.no_grad():
            model.eval()
            info = trainprocess(devloader, model, False, lossFunc, optimizer)
        errlist = info['errlist']
        meanDevLoss = np.mean(info['losslist'])
        devAcc = np.mean(info['rightlist'])
        printInfo(args, epoch, meanTrainLoss, trainAcc, meanDevLoss, devAcc, errlist)
        record['acc'].append(devAcc)
    print('max acc:', np.max(record['acc']))

def parseArgs():
    parser = argparse.ArgumentParser(description='HTrans')
    parser.add_argument('--dataset', type=str, default='./datasets',
                        help='directory of dataset')
    parser.add_argument('--epoch', type=int, default=200,
                        help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4,
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
