import numpy as np
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import time
from patches import generate_iter
from dataset import load_dataset, split_traintest, apply_pca
import os
import logging
import argparse
import sys
import random
import datetime
from model.IDNet import IDnet
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv


# 计算每个类和平均精度
def report_AA_CA(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

# 生成分类报告并评估指标
def report_metrics(y_test, y_pred_test):
    classification = classification_report(y_test, y_pred_test, digits=4)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = report_AA_CA(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100


class OptInit:
    def __init__(self):
        parser = argparse.ArgumentParser(description="HSI & Lidar Classification")

        # dataset
        parser.add_argument("--dataset_name", default=r"muufl", help="augsburg, muufl, houston, trento, yancheng")  # todo
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--pca_components", type=int, default=20)
        parser.add_argument("--hsi_channels", default=20, type=int, help="channels of hsi")
        parser.add_argument("--lidar_channels", default=1, type=int, help="channels of lidar")
        parser.add_argument("--patch_size", default=12, type=int, help="channels of hsi")
        parser.add_argument("--num_experts", default=10, type=int)
        parser.add_argument("--k", default=10, type=int)
        parser.add_argument("--dim",default=16,type=int)

        # optimizer
        parser.add_argument("--epochs", default=1000, type=int)
        parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
        parser.add_argument("--step_size", default=50, type=float, help="step_size")
        parser.add_argument("--gamma", default=0.9, type=float, help="gamma")
        # misc
        parser.add_argument("--work_dirs", default="modeltu", help="experiment results saved at work_dirs")
        parser.add_argument("--seed", default=0, type=int)
        self.args = parser.parse_args()
        self.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # dataset setting
        if self.args.dataset_name == "augsburg":
            self.args.num_classes = 7
            self.args.patch_size=12
        elif self.args.dataset_name == "muufl": 
            self.args.num_classes = 11
            self.args.patch_size=12
        elif self.args.dataset_name == "houston":
            self.args.num_classes = 15
            self.args.patch_size=8
        elif self.args.dataset_name == "trento":
            self.args.num_classes = 6
            self.args.patch_size=8
        
        self._set_seed(self.args.seed)
        self._configure_logger()
        self._print_args()

    def get_args(self):
        return self.args

    def _print_args(self):
        self.args.logger.info("*******************       start  args      *******************")
        for arg, content in self.args.__dict__.items():
            self.args.logger.info("{}:{}".format(arg, content))
        self.args.logger.info("*******************        end   args     *******************")

    def _configure_logger(self):
        logger = logging.getLogger(name="logger")
        logger.setLevel("DEBUG")

        date_time = datetime.datetime.now()
        time = date_time.strftime("%Y%m%d_%H%M%S")
        # print(time)
        self.args.time = time
        # experiment dir name
        experiment_name = "IDNet2_" + self.args.dataset_name
        self.args.log_file_name = experiment_name + ".log"
        self.args.work_dirs = os.path.join(self.args.work_dirs, experiment_name, time)
        os.makedirs(self.args.work_dirs)
        log_file = os.path.join(self.args.work_dirs, self.args.log_file_name)  # TODO
        file_handler = logging.FileHandler(log_file)
        stdout_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(fmt="%(asctime)s| %(message)s")
        file_handler.setLevel("INFO")
        file_handler.setFormatter(formatter)
        stdout_handler.setLevel("DEBUG")
        stdout_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)

        logger.info("=" * 88)
        logger.info("experiment log file saved at {}".format(self.args.work_dirs))
        self.args.logger = logger

    def _set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_data_loader(args):

    hsi, lidar, y = load_dataset(args)

    patch_size = args.patch_size
    patch_padding = int(patch_size // 2)
    logger = args.logger

    logger.info("hsi data shape: {}".format(hsi.shape))
    logger.info("Lidar data shape:{}".format(lidar.shape))
    logger.info("Label shape:{}".format(y.shape))
    # pca
    logger.info("\n... ... PCA tranformation ... ...")
    hsi = apply_pca(hsi, args.pca_components)
    logger.info("Data shape after PCA: {}".format(hsi.shape))

    hsi_all_data = hsi.reshape(np.prod(hsi.shape[:2]), np.prod(hsi.shape[2:]))
    lidar_all_data = lidar.reshape(
        np.prod(lidar.shape[:2]),
    )
    gt = y.reshape(
        np.prod(y.shape[:2]),
    )
    gt = gt.astype(np.int_)

    assert args.num_classes == max(gt)

    logger.info("num_classes = {}".format(args.num_classes))
    # 归一化
    hsi_all_data = preprocessing.scale(hsi_all_data)
    hsi_data = hsi_all_data.reshape(hsi.shape[0], hsi.shape[1], hsi.shape[2])
    whole_data_hsi = hsi_data
    # padding
    padded_data_hsi = np.pad(
        whole_data_hsi,
        ((patch_padding, patch_padding), (patch_padding, patch_padding), (0, 0)),
        "constant",
        constant_values=0,
    )

    lidar_all_data = preprocessing.scale(lidar_all_data)
    lidar_data = lidar_all_data.reshape(lidar.shape[0], lidar.shape[1])
    whole_data_lidar = lidar_data
    padded_data_lidar = np.pad(
        whole_data_lidar,
        ((patch_padding, patch_padding), (patch_padding, patch_padding)),
        "constant",
        constant_values=0,
    )
    logger.info("\n... ... create train & test data ... ...")
    # 用gt划分
    train_indices, test_indices = split_traintest(args, gt)
    train_indices = np.array(train_indices)
    print(train_indices.shape)
    # train_indices, test_indices = sampling(0.99, gt)
    total_samples = len(train_indices) + len(test_indices)
    train_samples = len(train_indices)
    logger.info("Train size:{}".format(train_samples))
    test_samples = total_samples - train_samples
    logger.info("Test size: {}".format(test_samples))

    logger.info("\n-----Selecting Small Cube from the Original Cube Data-----")
    train_iter, test_iter = generate_iter(
        train_samples,
        train_indices,
        test_samples,
        test_indices,
        whole_data_hsi,
        whole_data_lidar,
        patch_padding,
        padded_data_hsi,
        padded_data_lidar,
        gt,
        args,
    )
    
    return train_iter, test_iter


# 测试模型，提取预测结果
def test(model, test_loader, args):
    count = 0
    # 模型测试
    model.eval()
    y_pred_test = 0
    y_test = 0
    for hsi_data, lidar_data, labels in test_loader:
        hsi_data, lidar_data = hsi_data.to(args.device), lidar_data.to(args.device)
        output = model(hsi_data, lidar_data)
        output = np.argmax(output.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = output
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, output))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test


def main():
    args = OptInit().get_args()
    
    start = time.time()
    
    train_loader, test_iter = create_data_loader(args)
    
    model = IDnet(classes=args.num_classes,dim=args.dim, patch=args.patch_size, num_experts=args.num_experts,k=args.k,
                            hsi_inchannel=args.pca_components).to(args.device)
    
    print(args.dataset_name.lower())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    total_loss = 0
    best_loss = 100.0

    tic1 = time.perf_counter()
    best_loss = [float('inf'),float('inf'),float('inf')]
    best_model_paths = ''
    for epoch in range(args.epochs):
        model.train()
        for j, (hsi_data, lidar_data, label) in enumerate(train_loader):
            hsi_data, lidar_data, label = (
                hsi_data.to(args.device),
                lidar_data.to(args.device),
                label.to(args.device),
            )
            
            output = model(hsi_data, lidar_data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()

            clip_value = 1.0
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            total_loss += loss.item()
        args.logger.info(
            "[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]"
            % (epoch + 1, total_loss / (epoch + 1), loss.item())
        )
        lr_scheduler.step()
        if loss.item() < best_loss[0]:
            best_loss[2] = best_loss[1]
            best_loss[1] = best_loss[0]
            best_loss[0] = loss.item()
            torch.save(model.state_dict(), f"{args.work_dirs}/_best_model.pth")
            
        elif loss.item() < best_loss[1]:
            best_loss[2] = best_loss[1]
            best_loss[1] = loss.item()

        elif loss.item() < best_loss[2]:
            best_loss[2] = loss.item()
    #model.save('my_model.pkl')
    
    args.logger.info("Finished Training")
    toc1 = time.perf_counter()


    # test
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(model, test_iter, args)
    toc2 = time.perf_counter()
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = report_metrics(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    args.logger.info("{} Training_Time (s)".format(Training_Time))
    args.logger.info("{} Test_time (s)".format(Test_time))
    
    args.logger.info("\n{}".format(classification))
    args.logger.info("\n{}".format(confusion))

    args.logger.info("{} Overall accuracy (%)".format(oa))
    args.logger.info("{} Average accuracy (%)".format(aa))
    args.logger.info("{} Kappa accuracy (%)".format(kappa))
    args.logger.info("\n{} Each accuracy (%)".format(each_acc))
    end = time.time()
    args.logger.info("Total running time: {:.2f} s".format(end - start))


if __name__ == "__main__":
    main()

