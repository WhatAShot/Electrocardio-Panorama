import math
import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, accuracy_score, f1_score


def PSNR(pred, gt, rois=None, shave_border=0):
    psrn_all = []

    for i in range(pred.shape[0]):
        end_point = rois[i, -1, 0] if rois is not None else pred.shape[2]
        for j in range(pred.shape[1]):
            pred_single, gt_single = pred[i, j, :end_point], gt[i, j, :end_point]
            imdff = pred_single - gt_single
            rmse = math.sqrt(np.mean(imdff ** 2))
            if rmse == 0:
                psrn_all.append(100)
            else:
                psrn_all.append(20 * np.log10(1.0 / rmse))

    return np.mean(psrn_all)


def SSIM(pred, gt, rois=None):
    ssim_all = []
    for i in range(pred.shape[0]):
        end_point = rois[i, -1, 0] if rois is not None else pred.shape[2]
        for j in range(pred.shape[1]):
            ssim = structural_similarity(pred[i, j, :end_point], gt[i, j, :end_point], data_range=1.0)
            ssim_all.append(ssim)

    return np.mean(ssim_all)


def compute_clf_metrics(pred_labels, gt_labels, target_label=-1):
    pr_auc_list = []
    label_list = np.unique(gt_labels)
    label_list.sort()
    target_recall = None
    target_precision = None
    for label in label_list:
        input_gt_labels = np.where(gt_labels == label, 1, 0)
        input_pred_probs = pred_labels[:, label]
        precision, recall, _thresholds = precision_recall_curve(input_gt_labels, input_pred_probs)
        area = auc(recall, precision)
        pr_auc_list.append(area)
        if label == target_label:
            target_recall = recall
            target_precision = precision

    pred_labels = np.argmax(pred_labels, axis=1)
    print(np.sum(gt_labels == 0), np.sum(gt_labels == 1), np.sum(gt_labels == 2), np.sum(gt_labels == 3))
    precision = precision_score(gt_labels, pred_labels, average=None)
    recall = recall_score(gt_labels, pred_labels, average=None)
    acc = accuracy_score(gt_labels, pred_labels)
    result = {
        'mean_auc': np.mean(pr_auc_list),
        'acc': acc,
        'N_auc': pr_auc_list[0],
        'S_auc': pr_auc_list[1],
        'V_auc': pr_auc_list[2],
        'F_auc': pr_auc_list[3],
        'target_recall_points': target_recall,
        'target_precision_points': target_precision,
        'target_recall': recall[target_label],
        'target_precision': precision[target_label],

    }
    return result
