import argparse
import numpy as np
from model import HJ_Model
import dataloader
import torch
torch.set_printoptions(profile='full')
LABEL={'TjFree': [1., 0.], 
       'TjIn': [0., 1.]}

def isRight(pred: torch.Tensor, label: torch.Tensor) -> list:
    """
    Input: prediction, label
    Output: TP, FP, FN
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    if torch.argmax(label, dim=1) == 1 and torch.argmax(pred, dim=1) == 1:
        tp = 1
    elif torch.argmax(label, dim=1) == 0 and torch.argmax(pred, dim=1) == 1:
        fp = 1
    elif torch.argmax(label, dim=1) == 1 and torch.argmax(pred, dim=1) == 0:
        fn = 1
    else:
        tn = 1
    return [tp, fp, fn, tn]

def cal_metric(tp, fp, fn):
    precision = tp / (tp + fp + 1e-20)
    recall = tp / (tp + fn + 1e-20)
    F1score = 2 * precision * recall / (precision + recall + 1e-20)
    return [precision, recall, F1score]

def printInfo(args, epoch, trainLoss, trainAccRate, devLoss, devAccRate, errlist):
    print('epoch:', epoch, '/', args.epoch, end=' ')
    print('train loss:', round(trainLoss, 4), end=' ')
    print('train F1 score:', round(trainAccRate * 100, 1), '%')
    print('train loss:', round(devLoss, 4), end=' ')
    print('dev F1 score:', round(devAccRate * 100, 1), '%')
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
        label = torch.tensor(LABEL[ast.label])
        label = torch.reshape(label, (1, 2))
        loss = lossFunc(out, label)
        optimizer.zero_grad()
        if backward:
            loss.backward()
            optimizer.step()
        losslist.append(float(loss))
        right = isRight(out, label)
        rightlist.append(right)
        if right[1] == 1 or right[2] == 1:
            errlist.append(ast.name + ast.label) 
    info = {'losslist':losslist, 'rightlist':rightlist, 'errlist': errlist}
    return info

def train(args):
    recipeloader = dataloader.RecipeLoader(args.dataset)
    astloader = dataloader.ASTLoader(recipeloader)
    (trainloader, devloader) = dataloader.divide(astloader, args.devname)
    print('trainset len:', len(trainloader), 'devset len:', len(devloader))
    model = HJ_Model(5, 2)
    lossFunc = torch.nn.CrossEntropyLoss()
    # lossFunc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
    # start training
    record = {'meanTrainLoss':[], 'F1 socre':[]}
    for epoch in range(0, args.epoch):
        # train
        model.train()
        info = trainprocess(trainloader, model, True, lossFunc, optimizer)
        meanTrainLoss = np.mean(info['losslist'])
        [tp, tr, tf] = cal_metric(np.sum([m[0] for m in info['rightlist']]), 
                                  np.sum([m[1] for m in info['rightlist']]), 
                                  np.sum([m[2] for m in info['rightlist']]))
        # dev
        with torch.no_grad():
            model.eval()
            info = trainprocess(devloader, model, False, lossFunc, optimizer)
        errlist = info['errlist']
        meanDevLoss = np.mean(info['losslist'])
        [dp, dr, df] = cal_metric(np.sum([m[0] for m in info['rightlist']]), 
                                  np.sum([m[1] for m in info['rightlist']]), 
                                  np.sum([m[2] for m in info['rightlist']]))
        printInfo(args, epoch, meanTrainLoss, tf, meanDevLoss, df, errlist)
        record['F1 socre'].append(df)
    print('max F1 score:', np.max(record['F1 socre']))

def parseArgs():
    parser = argparse.ArgumentParser(description='HTrans')
    parser.add_argument('--dataset', type=str, default='./datasets',
                        help='directory of dataset')
    parser.add_argument('--epoch', type=int, default=200,
                        help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-5,
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
