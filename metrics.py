import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

def calculate_metrics(output, target):
    b = output.size(0)
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(-1).data.cpu().numpy()
    pred = output > 0.5
    gt = target > 0.5
    
    hd_pred = np.reshape(pred.astype(int),(b,-1))
    hd_gt = np.reshape(gt.astype(int),(b,-1))    
    hd = hausdorff_distance(hd_pred, hd_gt, distance="euclidean")
    
    confusion = confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    auc = roc_auc_score(gt, pred)
    a = (TP + TN) / float(TP + TN + FP + FN)  #accuracy
    r = TP / float(TP + FN) #Sensitivity
    s = TN / float(TN + FP)  #Specificity
    p = (TP+smooth) / (float(TP + FP)+smooth) #precision
    f1 = (2*p*r + smooth)/(p+r + smooth)
    
    #IOU score    
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    iou = (intersection + smooth) / (union + smooth)
    
    #Dice score
#    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    dice = (2. * (output*target).sum() + smooth) / ((output ** 2).sum() + (target ** 2).sum() + smooth)
    
    return {"iou":iou, "dice":dice, "accuracy":a, "precision":p, \
            "recall":r, "f1":f1, "specificity":s, "auc":auc, "hd":hd}
