import json

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import re
import torch
from tqdm import tqdm
from sklearn import metrics

from sklearn.metrics import auc

'''
calculate each rate
'''


def cal_rate(result, num, thres):
    all_number = len(result[0])
    # print all_number
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for item in range(all_number):
        class_prob = result[0][item, num]
        if class_prob >= thres:
            class_prob = 1
        if class_prob == 1:
            if result[1][item, num] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if result[1][item, num] == 0:
                TN += 1
            else:
                FN += 1
    # print TP+FP+TN+FN
    accracy = float(TP + FP) / float(all_number)
    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP + FP)
    TPR = float(TP) / float(TP + FN)
    TNR = float(TN) / float(FP + TN)
    FNR = float(FN) / float(TP + FN)
    FPR = float(FP) / float(FP + TN)
    # print accracy, precision, TPR, TNR, FNR, FPR
    return accracy, precision, TPR, TNR, FNR, FPR


def get_scores(predicts, labels):
    assert predicts.shape == labels.shape
    class_num = predicts.shape[-1]
    res = []
    print('Start calculating...')
    for i in range(class_num):
        print('Calculating class{}...'.format(i))
        threshold_vaule = sorted(predicts[:, i])
        threshold_num = len(threshold_vaule)
        accracy_array = np.zeros(threshold_num)
        precision_array = np.zeros(threshold_num)
        TPR_array = np.zeros(threshold_num)
        TNR_array = np.zeros(threshold_num)
        FNR_array = np.zeros(threshold_num)
        FPR_array = np.zeros(threshold_num)
        # calculate all the rates
        for thres in tqdm(range(threshold_num)):
            accracy, precision, TPR, TNR, FNR, FPR = cal_rate((predicts, labels), i, threshold_vaule[thres])
            accracy_array[thres] = accracy
            precision_array[thres] = precision
            TPR_array[thres] = TPR
            TNR_array[thres] = TNR
            FNR_array[thres] = FNR
            FPR_array[thres] = FPR
        # print TPR_array
        # print FPR_array
        AUC = np.trapz(TPR_array, FPR_array)
        threshold = np.argmin(abs(FNR_array - FPR_array))
        EER = (FNR_array[threshold] + FPR_array[threshold]) / 2

        res_dict = {'TPR_array': TPR_array, 'FPR_array': FPR_array, 'accuracy': accracy_array[threshold],
                    'AUC': AUC,
                    'EER': EER}
        res.append(res_dict)
    return res


def get_acc(preds, labs):
    preds_res = preds >= 0.5
    right_count = 0
    for i in range(len(preds_res)):
        if preds_res[i] == labs[i]:
            right_count += 1
    return right_count / len(preds_res)


def compute_eer(fpr, tpr, threshold):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


def get_scores_for_binary(predictions, labels, positive_class=1):
    preds = predictions[:, positive_class]
    labs = labels[:, positive_class]

    fpr, tpr, thresholds = metrics.roc_curve(labs, preds, pos_label=1)
    acc = get_acc(preds, labs)
    AUC = auc(fpr, tpr)
    EER = compute_eer(fpr, tpr, thresholds)
    res_dict = {'TPR_array': tpr, 'FPR_array': fpr, 'accuracy': float(acc),
                'AUC': float(AUC),
                'EER': float(EER)}
    return res_dict


def do_face_evaluation(dataset, predictions, output_folder, logger, visualize_scores=True):
    pred_class_tensor_list = []
    class_label_tensor_list = []
    for image_id, prediction in enumerate(predictions):
        img_path, class_label, domain_label = dataset.data_list[image_id]
        pred_class_tensor = prediction[:, :2]
        pred_domain_tensor = prediction[:, 2:]
        class_label_tensor = torch.zeros_like(pred_class_tensor)
        class_label_tensor[:, class_label] += 1
        pred_class_tensor_list.append(pred_class_tensor)
        class_label_tensor_list.append(class_label_tensor)
    preds_arr = torch.cat(pred_class_tensor_list, dim=0).numpy()
    labels_arr = torch.cat(class_label_tensor_list, dim=0).numpy()

    if preds_arr.shape[-1] == 2:
        res = get_scores_for_binary(preds_arr, labels_arr)
    else:
        res = get_scores(preds_arr, labels_arr)[-1]
    TPR_array = res.pop('TPR_array')
    FPR_array = res.pop('FPR_array')
    logger.info('Accuracy: {}, AUC: {}, EER: {}'.format(res['accuracy'], res['AUC'],
                                                        res['EER']))
    if output_folder is not None:
        res_output_path = os.path.join(output_folder, 'eval_result.json')
        with open(res_output_path, 'w', encoding='utf-8') as f:
            json.dump(res, f)
        if visualize_scores:
            vis_output_path = os.path.join(output_folder, 'eval_roc.png')
            plt.plot(FPR_array, TPR_array)
            plt.title('roc')
            plt.xlabel('FPR_array')
            plt.ylabel('TPR_array')
            plt.legend()
            plt.savefig(vis_output_path)

# disease_class = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
#                  'Pneumothorax']
# style = ['r-', 'g-', 'b-', 'y-', 'r--', 'g--', 'b--', 'y--']
# '''
# plot roc and calculate AUC/ERR, result: (prob, label)
# '''
# prob = np.random.rand(100, 8)
# print(prob)
# label = np.where(prob >= 0.5, prob, 0)
# label = np.where(label < 0.5, label, 1)
# count = np.count_nonzero(label)
# label = np.zeros((100, 8))
# label[1:20, :] = 1
# for clss in range(len(disease_class)):
#     threshold_vaule = sorted(prob[:, clss])
#     threshold_num = len(threshold_vaule)
#     accracy_array = np.zeros(threshold_num)
#     precision_array = np.zeros(threshold_num)
#     TPR_array = np.zeros(threshold_num)
#     TNR_array = np.zeros(threshold_num)
#     FNR_array = np.zeros(threshold_num)
#     FPR_array = np.zeros(threshold_num)
#     # calculate all the rates
#     for thres in range(threshold_num):
#         accracy, precision, TPR, TNR, FNR, FPR = cal_rate((prob, label), clss, threshold_vaule[thres])
#         accracy_array[thres] = accracy
#         precision_array[thres] = precision
#         TPR_array[thres] = TPR
#         TNR_array[thres] = TNR
#         FNR_array[thres] = FNR
#         FPR_array[thres] = FPR
#     # print TPR_array
#     # print FPR_array
#     AUC = np.trapz(TPR_array, FPR_array)
#     threshold = np.argmin(abs(FNR_array - FPR_array))
#     EER = (FNR_array[threshold] + FPR_array[threshold]) / 2
#     print('disease %10s threshold : %f' % (disease_class[clss], threshold))
#     print('disease %10s accuracy : %f' % (disease_class[clss], accracy_array[threshold]))
#     print('disease %10s EER : %f AUC : %f' % (disease_class[clss], EER, -AUC))
#     plt.plot(FPR_array, TPR_array, style[clss], label=disease_class[clss])
# plt.title('roc')
# plt.xlabel('FPR_array')
# plt.ylabel('TPR_array')
# plt.legend()
# plt.show()
