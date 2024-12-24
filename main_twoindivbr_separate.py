import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import warnings

warnings.filterwarnings('ignore')

import math
from dataset_prep import Dataset_Train, Dataset_Test
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import confusion_matrix
import random
from sklearn.metrics import roc_auc_score
from model_attloss_nosumas1_twoindivbr import *
import util
import os
import linecache
import re


def step_decay(epoch, learning_rate, drop, epochs_drop):
    """
    learning rate step decay
    :param epoch: current training epoch
    :param learning_rate: initial learning rate
    :return: learning rate after step decay
    """
    initial_lrate = learning_rate
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def train(epoch, model, learning_rate, train_loader, drop, epochs_drop, space, csatt, CA_MOD, attcoef, indivcoef_1,
          indivcoef_2):
    log_interval = 1
    LEARNING_RATE = step_decay(epoch, learning_rate, drop, epochs_drop)
    print(f'Learning Rate: {LEARNING_RATE}')

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    clf_criterion = nn.CrossEntropyLoss()

    model.train()

    train_correct = 0
    train_loss_accumulate = 0.0

    TN = 0
    FP = 0
    FN = 0
    TP = 0

    tr_auc_y_gt = []
    tr_auc_y_pred = []

    len_dataloader = len(train_loader)
    for step, source_sample_batch in enumerate(train_loader):
        ################################ extract sMRI deep features ################################
        if space == 'brain':
            sMRI_deep = source_sample_batch['sMRI_deep_brainspace']
        elif space == 'mni':
            sMRI_deep = source_sample_batch['sMRI_deep_mnispace']
        sMRI_deep = sMRI_deep.unsqueeze(1)
        sMRI_deep = sMRI_deep / 255.0

        ################################ extract sMRI trad features ################################
        sMRI_trad = source_sample_batch['sMRI_trad']

        ################################ extract fMRI features ################################
        dyn_a, sampling_points = util.bold.process_dynamic_fc(source_sample_batch['fMRI'], window_size=Window_Size,
                                                              window_stride=Window_Stride,
                                                              dynamic_length=Dynamic_Length)

        if step == 0:
            dyn_v = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points),
                           b=BATCH_SIZE)  # dyn_v (bs, segment_num, 116, 116)
        if not dyn_v.shape[1] == dyn_a.shape[1]:
            dyn_v = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=BATCH_SIZE)
        if len(dyn_a) < BATCH_SIZE:
            dyn_v = dyn_v[:len(dyn_a)]

        label = source_sample_batch['label']

        ####### model training ########
        logit, loss_ca, logit_sMRIdeeptrad, logit_fMRI = model(dyn_v.to(device), dyn_a.to(device), sMRI_trad.to(device),
                                                               sMRI_deep.to(device), csatt, CA_MOD)
        loss_clf = clf_criterion(logit, label.to(device))
        loss_clf_sMRIdeeptrad = clf_criterion(logit_sMRIdeeptrad, label.to(device))
        loss_clf_fMRI = clf_criterion(logit_fMRI, label.to(device))

        loss = loss_clf + attcoef * loss_ca + indivcoef_1 * loss_clf_sMRIdeeptrad + indivcoef_2 * loss_clf_fMRI

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_pred = logit.argmax(1)
        train_prob = logit.softmax(1)[:, 1]  # prob
        train_loss_accumulate += loss

        train_correct += sum(train_pred.data.cpu().numpy() == label.cpu().numpy())

        TN_tmp, FP_tmp, FN_tmp, TP_tmp = confusion_matrix(label.cpu().numpy(), train_pred.data.cpu().numpy(),
                                                          labels=[0, 1]).ravel()
        TN += TN_tmp
        FP += FP_tmp
        FN += FN_tmp
        TP += TP_tmp
        tr_auc_y_gt.extend(label.cpu().numpy())
        tr_auc_y_pred.extend(train_prob.detach().cpu().numpy())

        if (step + 1) % log_interval == 0:
            print(
                "Train Epoch [{:4d}/{}] Step [{:2d}/{}]: loss_clf={:.6f} loss_ca={:.6f} loss_clf_sMRIdeeptrad={:.6f} loss_clf_fMRI={:.6f}".format(
                    epoch, TRAIN_EPOCHS, step + 1, len_dataloader, loss_clf.data, loss_ca.data,
                    loss_clf_sMRIdeeptrad.data, loss_clf_fMRI.data))

    train_loss_accumulate /= len_dataloader
    train_acc = (TP + TN) / (TP + FP + TN + FN)  # accuracy of each class
    train_AUC = roc_auc_score(tr_auc_y_gt, tr_auc_y_pred)
    train_F1 = (2 * TP) / (2 * TP + FP + FN)

    print(
        'Train set: Average classification loss: {:.4f}, Accuracy: {}/{} ({:.4f}), train_AUC: {:.5}, train_F1: {:.5}'.format(
            train_loss_accumulate, train_correct, (len_dataloader * BATCH_SIZE), train_acc, train_AUC, train_F1))

    # save checkpoint.pth, save train loss and acc to a txt file
    if epoch == 120:
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'params_lr' + str(learning_rate) + '_ep' + str(
            epochs_drop) + '_drop' + str(drop) + '_Space' + str(space) + '_' + str(csatt) + '_' + str(
            CA_MOD) + '_' + str(attcoef) + '_' + str(indivcoef_1) + '_' + str(indivcoef_2), task,
                                                    'fold_' + str(fold) + '_epoch_' + str(epoch) + '.pth'))
    with open(os.path.join(SAVE_PATH, 'params_lr' + str(learning_rate) + '_ep' + str(epochs_drop) + '_drop' + str(
            drop) + '_Space' + str(space) + '_' + str(csatt) + '_' + str(CA_MOD) + '_' + str(attcoef) + '_' + str(
            indivcoef_1) + '_' + str(indivcoef_2), task, 'fold_' + str(fold) + '_train_loss_and_acc.txt'), 'a') as f:
        f.write('epoch {}, total loss {:.5}, train acc {:.5}, train_AUC {:.5}, train_F1: {:.5}\n'.format(epoch,
                                                                                                         train_loss_accumulate,
                                                                                                         train_acc,
                                                                                                         train_AUC,
                                                                                                         train_F1))


def test_1(model, target_loader, space, csatt, CA_MOD) -> object:
    """
    :param model: trained alexnet on source data set
    :param target_loader: target dataloader
    :return: correct num
    """

    with torch.no_grad():
        model.eval()
        test_correct = 0
        TN = 0
        FP = 0
        FN = 0
        TP = 0

        te_auc_y_gt = []
        te_auc_y_pred = []

        for step, test_sample_batch in enumerate(target_loader):

            ################################ extract sMRI deep features ################################
            if space == 'brain':
                sMRI_deep = test_sample_batch['sMRI_deep_brainspace']
            elif space == 'mni':
                sMRI_deep = test_sample_batch['sMRI_deep_mnispace']
            sMRI_deep = sMRI_deep.unsqueeze(1)
            sMRI_deep = sMRI_deep / 255.0

            ################################ extract sMRI trad features ################################
            sMRI_trad = test_sample_batch['sMRI_trad']

            ################################ extract fMRI features ################################
            dyn_a, sampling_points = util.bold.process_dynamic_fc(test_sample_batch['fMRI'], window_size=Window_Size,
                                                                  window_stride=Window_Stride,
                                                                  dynamic_length=Dynamic_Length)

            if step == 0:
                dyn_v = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points),
                               b=BATCH_SIZE)  # dyn_v (bs, segment_num, 116, 116)
            if not dyn_v.shape[1] == dyn_a.shape[1]:
                dyn_v = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=BATCH_SIZE)
            if len(dyn_a) < BATCH_SIZE:
                dyn_v = dyn_v[:len(dyn_a)]

            logit, _, logit_sMRIdeeptrad, logit_fMRI = model(dyn_v.to(device), dyn_a.to(device), sMRI_trad.to(device),
                                                             sMRI_deep.to(device), csatt, CA_MOD)
            label = test_sample_batch['label']

            test_pred = logit.argmax(1)
            test_prob = logit.softmax(1)[:, 1]  # prob

            test_correct += sum(test_pred.data.cpu().numpy() == label.cpu().numpy())
            TN_tmp, FP_tmp, FN_tmp, TP_tmp = confusion_matrix(label.cpu().numpy(), test_pred.data.cpu().numpy(),
                                                              labels=[0, 1]).ravel()
            TN += TN_tmp
            FP += FP_tmp
            FN += FN_tmp
            TP += TP_tmp
            te_auc_y_gt.extend(label.cpu().numpy())
            te_auc_y_pred.extend(test_prob.detach().cpu().numpy())

        TPR = TP / (TP + FN)  # Sensitivity/ hit rate/ recall/ true positive rate
        TNR = TN / (TN + FP)  # Specificity/ true negative rate
        PPV = TP / (TP + FP)  # Precision/ positive predictive value
        test_acc = (TP + TN) / (TP + FP + TN + FN)  # accuracy of each class
        test_AUC = roc_auc_score(te_auc_y_gt, te_auc_y_pred)
        test_F1 = (2 * TP) / (2 * TP + FP + FN)

        print(
            'Test set: Correct_num: {}, test_acc: {:.4f}, test_AUC: {:.4f}, test_F1: {:.4f}, TPR: {:.4f}, TNR: {:.4f}, PPV:{:.4f}\n'.format(
                test_correct, test_acc, test_AUC, test_F1, TPR, TNR, PPV))

        # save test loss and acc to a txt file
        with open(os.path.join(SAVE_PATH, 'params_lr' + str(learning_rate) + '_ep' + str(epochs_drop) + '_drop' + str(
                drop) + '_Space' + str(space) + '_' + str(csatt) + '_' + str(CA_MOD) + '_' + str(attcoef) + '_' + str(
                indivcoef_1) + '_' + str(indivcoef_2), task, 'fold_' + str(fold) + '_test1_loss_and_acc.txt'),
                  'a') as f:
            f.write(
                'epoch {}, test_acc {:.5}, test_AUC {:.5}, test_F1: {:.4f}, TPR {:.5}, TNR {:.5}, PPV {:.5}\n'.format(
                    epoch, test_acc, test_AUC, test_F1, TPR, TNR, PPV))


if __name__ == '__main__':

    # ROOT_PATH = '/home/yuqifang/projects/MICCAI2024/Data/'
    # SAVE_PATH = '/home/yuqifang/projects/MICCAI2024/Experiments/ablation_study/ASFF/checkpoints_twoindivbr_separate/'

    ROOT_PATH = '/shenlab/lab_stor4/yuqifang/projects/MICCAI2024/Data/'
    SAVE_PATH = '/shenlab/lab_stor4/yuqifang/projects/MICCAI2024/Experiments/ablation_study/ASFF/checkpoints_twoindivbr_separate/'

    # hyperparameters tuning
    three_cls_task = ['ANI_control']  # ['ANI_control', 'noANI_control', 'ANI_noANI']

    TRAIN_EPOCHS = 150
    FOLD_NUM = 5
    BATCH_SIZE = 4
    learning_rate_lst = [0.0001]  # [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    epochs_drop_lst = [30.0]  # [10.0, 30.0]
    drop_lst = [0.5]
    space_lst = ['mni']  # ['brain', 'mni']
    method_lst = ['Simple3D']  # sMRI deep models. can add more methods
    csatt_lst = ['pair2']  # pair3 means cross-attention between any pairs of 3 modalities
    CA_MOD_lst = ['CA4']  # ['CA2', 'CA3', 'CA4', 'CA23', 'CA24', 'CA34', 'CA234']
    attcoef_lst = [1.0]
    indivcoef_lst_1 = [0.8]  # [0.0, 0.5, 1.0, 2.0]
    indivcoef_lst_2 = [0.8]  # [0.0, 0.5, 1.0, 2.0]
    Window_Size = 40
    Window_Stride = 30
    Dynamic_Length = 230
    train_pct = 0.7

    for task in three_cls_task:
        for learning_rate in learning_rate_lst:
            for epochs_drop in epochs_drop_lst:
                for drop in drop_lst:
                    for space in space_lst:
                        for method in method_lst:
                            for csatt in csatt_lst:
                                for CA_MOD in CA_MOD_lst:
                                    for attcoef in attcoef_lst:
                                        for indivcoef_1 in indivcoef_lst_1:
                                            for indivcoef_2 in indivcoef_lst_2:

                                                if not os.path.exists(os.path.join(SAVE_PATH, 'params_lr' + str(
                                                        learning_rate) + '_ep' + str(epochs_drop) + '_drop' + str(
                                                        drop) + '_Space' + str(space) + '_' + str(csatt) + '_' + str(
                                                        CA_MOD) + '_' + str(attcoef) + '_' + str(
                                                        indivcoef_1) + '_' + str(indivcoef_2), task)):
                                                    os.makedirs(os.path.join(SAVE_PATH, 'params_lr' + str(
                                                        learning_rate) + '_ep' + str(epochs_drop) + '_drop' + str(
                                                        drop) + '_Space' + str(space) + '_' + str(csatt) + '_' + str(
                                                        CA_MOD) + '_' + str(attcoef) + '_' + str(
                                                        indivcoef_1) + '_' + str(indivcoef_2), task))

                                                # load data
                                                cls1, cls0 = task.split('_')[0], task.split('_')[1]
                                                cls1_data_subjid = np.load(
                                                    os.path.join(ROOT_PATH, 'fuse_sfMRI_' + cls1 + '.npy'),
                                                    allow_pickle=True)
                                                cls1_label = np.ones(cls1_data_subjid.shape[0])
                                                cls0_data_subjid = np.load(
                                                    os.path.join(ROOT_PATH, 'fuse_sfMRI_' + cls0 + '.npy'),
                                                    allow_pickle=True)
                                                cls0_label = np.zeros(cls0_data_subjid.shape[0])

                                                for fold in range(FOLD_NUM):
                                                    print("fold:", fold)

                                                    seed = fold  # fold
                                                    np.random.seed(seed)
                                                    random.seed(seed)
                                                    torch.manual_seed(seed)  # cpu
                                                    torch.cuda.manual_seed(seed)
                                                    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
                                                    torch.backends.cudnn.benchmark = False
                                                    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致

                                                    # randomly divide training data (70%) and test data (30%)
                                                    tr_data_subjid, tr_label, te_data_subjid, te_label = util.data_proc.rand_div(
                                                        cls1_data_subjid, cls1_label, cls0_data_subjid, cls0_label,
                                                        train_pct)

                                                    # data loader
                                                    train_dataset = Dataset_Train(tr_data_subjid, tr_label)
                                                    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                                                              shuffle=True, num_workers=0,
                                                                              pin_memory=True, drop_last=True)

                                                    test_dataset = Dataset_Test(te_data_subjid, te_label)
                                                    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                                                             shuffle=False, num_workers=0,
                                                                             pin_memory=True, drop_last=False)

                                                    print('Construct model begin:')
                                                    # if method == 'Simple3D':
                                                    net_sfMRI = United_Model(fMRI_input_dim=116,
                                                                             sMRI_trad_input_dim=4858, hidden_dim=64,
                                                                             num_classes=2, num_heads=1, num_layers=2,
                                                                             sparsity=30, dropout=0.5, cls_token='sum',
                                                                             readout='sero')

                                                    device = torch.device(
                                                        "cuda:0" if torch.cuda.is_available() else "cpu")
                                                    print('device:', device)

                                                    net_sfMRI.to(device)

                                                    with open(os.path.join(SAVE_PATH, 'params_lr' + str(
                                                            learning_rate) + '_ep' + str(epochs_drop) + '_drop' + str(
                                                            drop) + '_Space' + str(space) + '_' + str(
                                                            csatt) + '_' + str(CA_MOD) + '_' + str(attcoef) + '_' + str(
                                                            indivcoef_1) + '_' + str(indivcoef_2), task, 'fold_' + str(
                                                            fold) + '_train_loss_and_acc.txt'), 'a') as f:
                                                        f.write(
                                                            'total_epoch: {}, batch_size: {}, initial_lr: {:.8}, drop_lr: {:.5}, drop_lr_per_epoch: {}\n'.format(
                                                                TRAIN_EPOCHS, BATCH_SIZE, learning_rate, drop,
                                                                epochs_drop))
                                                    with open(os.path.join(SAVE_PATH, 'params_lr' + str(
                                                            learning_rate) + '_ep' + str(epochs_drop) + '_drop' + str(
                                                            drop) + '_Space' + str(space) + '_' + str(
                                                            csatt) + '_' + str(CA_MOD) + '_' + str(attcoef) + '_' + str(
                                                            indivcoef_1) + '_' + str(indivcoef_2), task, 'fold_' + str(
                                                            fold) + '_test1_loss_and_acc.txt'), 'a') as f:
                                                        f.write(
                                                            'total_epoch: {}, batch_size: {}, initial_lr: {:.8}, drop_lr: {:.5}, drop_lr_per_epoch: {}\n'.format(
                                                                TRAIN_EPOCHS, BATCH_SIZE, learning_rate, drop,
                                                                epochs_drop))

                                                    for epoch in range(1, TRAIN_EPOCHS + 1):
                                                        print(f'Train Epoch {epoch}:')
                                                        train(epoch, net_sfMRI, learning_rate, train_loader, drop,
                                                              epochs_drop, space, csatt, CA_MOD, attcoef, indivcoef_1,
                                                              indivcoef_2)
                                                        correct_1 = test_1(net_sfMRI, test_loader, space, csatt, CA_MOD)


                                                # calculate the mean value of five folds
                                                def get_line_context(file_path, line_number=(TRAIN_EPOCHS + 1)):
                                                    return linecache.getline(file_path, line_number).strip()


                                                for line_num in range(2, TRAIN_EPOCHS + 2):
                                                    test_result_list = []
                                                    for fold in range(FOLD_NUM):
                                                        txt_file_path = os.path.join(SAVE_PATH, 'params_lr' + str(
                                                            learning_rate) + '_ep' + str(epochs_drop) + '_drop' + str(
                                                            drop) + '_Space' + str(space) + '_' + str(
                                                            csatt) + '_' + str(CA_MOD) + '_' + str(attcoef) + '_' + str(
                                                            indivcoef_1) + '_' + str(indivcoef_2), task, 'fold_' + str(
                                                            fold) + '_test1_loss_and_acc.txt')
                                                        test_result_str = get_line_context(txt_file_path,
                                                                                           line_number=line_num)
                                                        test_result_str = test_result_str.replace('nan', '10000')
                                                        test_result_str_numpart = re.findall(r"\d+\.?\d*",
                                                                                             test_result_str)  # only extract number in a str
                                                        test_result_str_numpart_float = []
                                                        for num in test_result_str_numpart:
                                                            test_result_str_numpart_float.append(float(num))

                                                        test_result_list.append(test_result_str_numpart_float)

                                                    test_acc_list = []
                                                    test_auc_list = []
                                                    test_f1_list = []
                                                    TPR_list = []
                                                    TNR_list = []
                                                    PPV_list = []

                                                    for repet_num in range(FOLD_NUM):
                                                        test_acc_list.append(test_result_list[repet_num][1])
                                                        test_auc_list.append(test_result_list[repet_num][2])
                                                        test_f1_list.append(
                                                            test_result_list[repet_num][4])  # [3] mean '1' in F1
                                                        TPR_list.append(test_result_list[repet_num][5])
                                                        TNR_list.append(test_result_list[repet_num][6])
                                                        PPV_list.append(test_result_list[repet_num][7])

                                                    # mean
                                                    test_acc_mean = np.mean(test_acc_list)
                                                    test_auc_mean = np.mean(test_auc_list)
                                                    test_f1_mean = np.mean(test_f1_list)
                                                    test_TPR_mean = np.mean(TPR_list)  # Sensitivity
                                                    test_TNR_mean = np.mean(TNR_list)  # Specificity
                                                    test_PPV_mean = np.mean(PPV_list)  # Precision

                                                    # std
                                                    test_acc_std = np.std(test_acc_list)
                                                    test_auc_std = np.std(test_auc_list)
                                                    test_f1_std = np.std(test_f1_list)
                                                    test_TPR_std = np.std(TPR_list)  # Sensitivity
                                                    test_TNR_std = np.std(TNR_list)  # Specificity
                                                    test_PPV_std = np.std(PPV_list)  # Precision

                                                    with open(os.path.join(SAVE_PATH, 'params_lr' + str(
                                                            learning_rate) + '_ep' + str(epochs_drop) + '_drop' + str(
                                                            drop) + '_Space' + str(space) + '_' + str(
                                                            csatt) + '_' + str(CA_MOD) + '_' + str(attcoef) + '_' + str(
                                                            indivcoef_1) + '_' + str(indivcoef_2), task,
                                                                           'a_mean_test1_acc_auc_params_lr' + str(
                                                                                   learning_rate) + '_ep' + str(
                                                                                   epochs_drop) + '_drop' + str(
                                                                                   drop) + '_Space' + str(
                                                                                   space) + '_' + str(
                                                                                   csatt) + '_' + str(
                                                                                   CA_MOD) + '_' + str(
                                                                                   attcoef) + '_' + str(
                                                                                   indivcoef_1) + '_' + str(
                                                                                   indivcoef_2) + '_' + str(
                                                                                   task) + '.txt'), 'a') as f:
                                                        f.write(
                                                            'epoch {}, test_ACC {}, test_AUC {}, test_F1 {}, test_TPR {}, test_TNR {}, test_PPV {}\n'.format(
                                                                (line_num - 1),
                                                                (str(format(100 * test_acc_mean, '.2f')) + '±' + str(
                                                                    format(100 * test_acc_std, '.2f'))),
                                                                (str(format(100 * test_auc_mean, '.2f')) + '±' + str(
                                                                    format(100 * test_auc_std, '.2f'))),
                                                                (str(format(100 * test_f1_mean, '.2f')) + '±' + str(
                                                                    format(100 * test_f1_std, '.2f'))),
                                                                (str(format(100 * test_TPR_mean, '.2f')) + '±' + str(
                                                                    format(100 * test_TPR_std, '.2f'))),
                                                                (str(format(100 * test_TNR_mean, '.2f')) + '±' + str(
                                                                    format(100 * test_TNR_std, '.2f'))),
                                                                (str(format(100 * test_PPV_mean, '.2f')) + '±' + str(
                                                                    format(100 * test_PPV_std, '.2f')))))
