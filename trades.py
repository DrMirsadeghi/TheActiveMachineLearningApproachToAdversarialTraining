import time
from collections import OrderedDict
import os, glob
import sys; sys.path.insert(0, '..')
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import mair
import torch

import torch.nn as nn
import torch.nn.functional as F
from mair.attacks import TPGD
from mair.attacks import PGD, CW, DeepFool, AutoAttack
from mair.attacks import PGD,FGSM
from mair.defenses.advtraining.advtrainer import AdvTrainer
from mair.defenses.advtraining.standard import Standard
from mair.defenses.advtraining.trades import TRADES

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.id_dict[img_path.split('/')[4].split('_')[0]]
        #label = self.id_dict[img_path.split('/')[4]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

def calc_entropy(input_tensor):
    lsm = nn.LogSoftmax()
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean()
    return entropy

class SIAutoAttack(AutoAttack):
  def __init__(
        self,
        model,
        norm="Linf",
        eps=8 / 255,
        version="standard",
        n_classes=10,
        seed=None,
        verbose=False,
    ):
    super().__init__( model)
    device = next(model.parameters()).device
    self.model = self.model.to(device)
  def save(
        self,
        data_loader,
        save_path=None,
        verbose=True,
        return_verbose=False,
        save_predictions=False,
        save_clean_inputs=False,
        save_type="float",
    ):
        r"""
        Save adversarial inputs as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_predictions (bool): True for saving predicted labels (Default: False)
            save_clean_inputs (bool): True for saving clean inputs (Default: False)

        """
        if save_path is not None:
            adv_input_list = []
            label_list = []
            if save_predictions:
                pred_list = []
            if save_clean_inputs:
                input_list = []

        correct = 0
        total = 0
        clean_correct= 0
        clean_total= 0
        l2_distance = []

        incorrect=0
        clean_incorrect=0
        success_correct= 0
        success_total= 0


        total_batch = len(data_loader)
        given_training = self.model.training

        for step, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            inputs=inputs.to(self.device)
            labels=labels.to(self.device)
            adv_inputs = self.__call__(inputs, labels)
            batch_size = len(inputs)

            if verbose or return_verbose:
                with torch.no_grad():
                    outputs = self.get_output_with_eval_nograd(adv_inputs)
                    clean_outputs = self.get_output_with_eval_nograd(inputs)

                    # Calculate robust accuracy
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = pred == labels.to(self.device)
                    correct += right_idx.sum()
                    incorrect += labels.size(0) - right_idx.sum()
                    rob_acc = 100 * float(correct) / total

                    # Calculate natural accuracy
                    _, pred = torch.max(clean_outputs.data, 1)
                    clean_total += labels.size(0)
                    right_idx = pred == labels.to(self.device)
                    clean_correct += right_idx.sum()
                    clean_incorrect += labels.size(0) - right_idx.sum()
                    acc = 100 * float(clean_correct) / total

                    # Calculate l2 distance
                    delta = (adv_inputs - inputs.to(self.device)).view(
                        batch_size, -1
                    )  # nopep8
                    l2_distance.append(
                        torch.norm(delta[~right_idx], p=2, dim=1)
                    )  # nopep8
                    l2 = torch.cat(l2_distance).mean().item()

                    ########################
                    # Calculate Success Rate
                    _, pred = torch.max(outputs.data, 1)
                    _, clean_pred = torch.max(clean_outputs.data, 1)
                    success_total += labels.size(0)
                    right_idx = (clean_pred == labels.to(self.device))*(pred != labels.to(self.device))
                    success_correct+= right_idx.sum()
                    success_rate = 100 * float(success_correct)/ clean_correct
                    # Calculate F1-robust
                    f1_robust = 100 * correct / (correct + 0.5*(incorrect + clean_incorrect))

                    # Calculate time computation
                    progress = (step + 1) / total_batch * 100
                    end = time.time()
                    elapsed_time = end - start

                    '''if verbose:
                        self._save_print(
                            progress, rob_acc, l2, elapsed_time, end="\r"
                        )  # nopep8'''

            if save_path is not None:
                adv_input_list.append(adv_inputs.detach().cpu())
                label_list.append(labels.detach().cpu())

                adv_input_list_cat = torch.cat(adv_input_list, 0)
                label_list_cat = torch.cat(label_list, 0)

                save_dict = {
                    "adv_inputs": adv_input_list_cat,
                    "labels": label_list_cat,
                }  # nopep8

                if save_predictions:
                    pred_list.append(pred.detach().cpu())
                    pred_list_cat = torch.cat(pred_list, 0)
                    save_dict["preds"] = pred_list_cat

                if save_clean_inputs:
                    input_list.append(inputs.detach().cpu())
                    input_list_cat = torch.cat(input_list, 0)
                    save_dict["clean_inputs"] = input_list_cat

                if self.normalization_used is not None:
                    save_dict["adv_inputs"] = self.inverse_normalize(
                        save_dict["adv_inputs"]
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.inverse_normalize(
                            save_dict["clean_inputs"]
                        )  # nopep8

                if save_type == "int":
                    save_dict["adv_inputs"] = self.to_type(
                        save_dict["adv_inputs"], "int"
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.to_type(
                            save_dict["clean_inputs"], "int"
                        )  # nopep8

                save_dict["save_type"] = save_type
                torch.save(save_dict, save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end="\n")

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time, success_rate, f1_robust

class SIDeepFool(DeepFool):
  def __init__(self, model, steps=50, overshoot=0.02):
    super().__init__( model)
    device = next(model.parameters()).device
    self.model = self.model.to(device)
  def save(
        self,
        data_loader,
        save_path=None,
        verbose=True,
        return_verbose=False,
        save_predictions=False,
        save_clean_inputs=False,
        save_type="float",
    ):
        r"""
        Save adversarial inputs as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_predictions (bool): True for saving predicted labels (Default: False)
            save_clean_inputs (bool): True for saving clean inputs (Default: False)

        """
        if save_path is not None:
            adv_input_list = []
            label_list = []
            if save_predictions:
                pred_list = []
            if save_clean_inputs:
                input_list = []

        correct = 0
        total = 0
        clean_correct= 0
        clean_total= 0
        l2_distance = []

        incorrect=0
        clean_incorrect=0

        success_correct= 0
        success_total= 0


        total_batch = len(data_loader)
        given_training = self.model.training

        for step, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            inputs=inputs.to(self.device)
            labels=labels.to(self.device)
            adv_inputs = self.__call__(inputs, labels)
            batch_size = len(inputs)

            if verbose or return_verbose:
                with torch.no_grad():
                    outputs = self.get_output_with_eval_nograd(adv_inputs)
                    clean_outputs = self.get_output_with_eval_nograd(inputs)

                    # Calculate robust accuracy
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = pred == labels.to(self.device)
                    correct += right_idx.sum()
                    incorrect += labels.size(0) - right_idx.sum()
                    rob_acc = 100 * float(correct) / total

                    # Calculate natural accuracy
                    _, pred = torch.max(clean_outputs.data, 1)
                    clean_total += labels.size(0)
                    right_idx = pred == labels.to(self.device)
                    clean_correct += right_idx.sum()
                    clean_incorrect += labels.size(0) - right_idx.sum()
                    acc = 100 * float(clean_correct) / total

                    # Calculate l2 distance
                    delta = (adv_inputs - inputs.to(self.device)).view(
                        batch_size, -1
                    )  # nopep8
                    l2_distance.append(
                        torch.norm(delta[~right_idx], p=2, dim=1)
                    )  # nopep8
                    l2 = torch.cat(l2_distance).mean().item()

                    ########################
                    # Calculate Success Rate
                    _, pred = torch.max(outputs.data, 1)
                    _, clean_pred = torch.max(clean_outputs.data, 1)
                    success_total += labels.size(0)
                    right_idx = (clean_pred == labels.to(self.device))*(pred != labels.to(self.device))
                    success_correct+= right_idx.sum()
                    success_rate = 100 * float(success_correct)/ clean_correct
                    # Calculate F1-robust
                    f1_robust = 100 * correct / (correct + 0.5*(incorrect + clean_incorrect))

                    # Calculate time computation
                    progress = (step + 1) / total_batch * 100
                    end = time.time()
                    elapsed_time = end - start

                    '''if verbose:
                        self._save_print(
                            progress, rob_acc, l2, elapsed_time, end="\r"
                        )  # nopep8'''

            if save_path is not None:
                adv_input_list.append(adv_inputs.detach().cpu())
                label_list.append(labels.detach().cpu())

                adv_input_list_cat = torch.cat(adv_input_list, 0)
                label_list_cat = torch.cat(label_list, 0)

                save_dict = {
                    "adv_inputs": adv_input_list_cat,
                    "labels": label_list_cat,
                }  # nopep8

                if save_predictions:
                    pred_list.append(pred.detach().cpu())
                    pred_list_cat = torch.cat(pred_list, 0)
                    save_dict["preds"] = pred_list_cat

                if save_clean_inputs:
                    input_list.append(inputs.detach().cpu())
                    input_list_cat = torch.cat(input_list, 0)
                    save_dict["clean_inputs"] = input_list_cat

                if self.normalization_used is not None:
                    save_dict["adv_inputs"] = self.inverse_normalize(
                        save_dict["adv_inputs"]
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.inverse_normalize(
                            save_dict["clean_inputs"]
                        )  # nopep8

                if save_type == "int":
                    save_dict["adv_inputs"] = self.to_type(
                        save_dict["adv_inputs"], "int"
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.to_type(
                            save_dict["clean_inputs"], "int"
                        )  # nopep8

                save_dict["save_type"] = save_type
                torch.save(save_dict, save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end="\n")

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time, success_rate, f1_robust

class SIPGD(PGD):
  def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
    super().__init__( model)
    device = next(model.parameters()).device
    self.model = self.model.to(device)
  def save(
        self,
        data_loader,
        save_path=None,
        verbose=True,
        return_verbose=False,
        save_predictions=False,
        save_clean_inputs=False,
        save_type="float",
    ):
        r"""
        Save adversarial inputs as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_predictions (bool): True for saving predicted labels (Default: False)
            save_clean_inputs (bool): True for saving clean inputs (Default: False)

        """
        if save_path is not None:
            adv_input_list = []
            label_list = []
            if save_predictions:
                pred_list = []
            if save_clean_inputs:
                input_list = []

        correct = 0
        total = 0
        clean_correct= 0
        clean_total= 0
        l2_distance = []

        incorrect =0
        clean_incorrect=0

        success_correct= 0
        success_total= 0

        total_batch = len(data_loader)
        given_training = self.model.training

        for step, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            inputs=inputs.to(self.device)
            labels=labels.to(self.device)
            adv_inputs = self.__call__(inputs, labels)
            batch_size = len(inputs)

            if verbose or return_verbose:
                with torch.no_grad():
                    outputs = self.get_output_with_eval_nograd(adv_inputs)
                    clean_outputs = self.get_output_with_eval_nograd(inputs)

                    # Calculate robust accuracy
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = pred == labels.to(self.device)
                    correct += right_idx.sum()
                    incorrect += labels.size(0) - right_idx.sum()
                    rob_acc = 100 * float(correct) / total

                    # Calculate natural accuracy
                    _, pred = torch.max(clean_outputs.data, 1)
                    clean_total += labels.size(0)
                    right_idx = pred == labels.to(self.device)
                    clean_correct += right_idx.sum()
                    clean_incorrect += labels.size(0) - right_idx.sum()
                    acc = 100 * float(clean_correct) / total

                    # Calculate l2 distance
                    delta = (adv_inputs - inputs.to(self.device)).view(
                        batch_size, -1
                    )  # nopep8
                    l2_distance.append(
                        torch.norm(delta[~right_idx], p=2, dim=1)
                    )  # nopep8
                    l2 = torch.cat(l2_distance).mean().item()

                    ########################
                    # Calculate Success Rate
                    _, pred = torch.max(outputs.data, 1)
                    _, clean_pred = torch.max(clean_outputs.data, 1)
                    success_total += labels.size(0)
                    right_idx = (clean_pred == labels.to(self.device)) * (pred != labels.to(self.device))
                    success_correct+= right_idx.sum()
                    success_rate = 100 * float(success_correct)/ clean_correct
                    # Calculate F1-robust
                    f1_robust = 100 * correct / (correct + 0.5*(incorrect + clean_incorrect))
                    
                    # Calculate time computation
                    progress = (step + 1) / total_batch * 100
                    end = time.time()
                    elapsed_time = end - start

                    '''if verbose:
                        self._save_print(
                            progress, rob_acc, l2, elapsed_time, end="\r"
                        )  # nopep8'''

            if save_path is not None:
                adv_input_list.append(adv_inputs.detach().cpu())
                label_list.append(labels.detach().cpu())

                adv_input_list_cat = torch.cat(adv_input_list, 0)
                label_list_cat = torch.cat(label_list, 0)

                save_dict = {
                    "adv_inputs": adv_input_list_cat,
                    "labels": label_list_cat,
                }  # nopep8

                if save_predictions:
                    pred_list.append(pred.detach().cpu())
                    pred_list_cat = torch.cat(pred_list, 0)
                    save_dict["preds"] = pred_list_cat

                if save_clean_inputs:
                    input_list.append(inputs.detach().cpu())
                    input_list_cat = torch.cat(input_list, 0)
                    save_dict["clean_inputs"] = input_list_cat

                if self.normalization_used is not None:
                    save_dict["adv_inputs"] = self.inverse_normalize(
                        save_dict["adv_inputs"]
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.inverse_normalize(
                            save_dict["clean_inputs"]
                        )  # nopep8

                if save_type == "int":
                    save_dict["adv_inputs"] = self.to_type(
                        save_dict["adv_inputs"], "int"
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.to_type(
                            save_dict["clean_inputs"], "int"
                        )  # nopep8

                save_dict["save_type"] = save_type
                torch.save(save_dict, save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end="\n")

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time, success_rate, f1_robust

class SIFGSM(FGSM):
  def __init__(self, model, eps=8 / 255):
        super().__init__(model, eps)
 
  def save(
        self,
        data_loader,
        save_path=None,
        verbose=True,
        return_verbose=False,
        save_predictions=False,
        save_clean_inputs=False,
        save_type="float",
    ):
        r"""
        Save adversarial inputs as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_predictions (bool): True for saving predicted labels (Default: False)
            save_clean_inputs (bool): True for saving clean inputs (Default: False)

        """
        if save_path is not None:
            adv_input_list = []
            label_list = []
            if save_predictions:
                pred_list = []
            if save_clean_inputs:
                input_list = []

        correct = 0
        total = 0
        clean_correct= 0
        clean_total= 0
        l2_distance = []

        incorrect =0
        clean_incorrect=0

        success_correct= 0
        success_total= 0

        total_batch = len(data_loader)
        given_training = self.model.training

        for step, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            inputs=inputs.to(self.device)
            labels=labels.to(self.device)
            adv_inputs = self.__call__(inputs, labels)
            batch_size = len(inputs)

            if verbose or return_verbose:
                with torch.no_grad():
                    outputs = self.get_output_with_eval_nograd(adv_inputs)
                    clean_outputs = self.get_output_with_eval_nograd(inputs)

                    # Calculate robust accuracy
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = pred == labels.to(self.device)
                    correct += right_idx.sum()
                    incorrect += labels.size(0) - right_idx.sum()
                    rob_acc = 100 * float(correct) / total

                    # Calculate natural accuracy
                    _, pred = torch.max(clean_outputs.data, 1)
                    clean_total += labels.size(0)
                    right_idx = pred == labels.to(self.device)
                    clean_correct += right_idx.sum()
                    clean_incorrect += labels.size(0) - right_idx.sum()
                    acc = 100 * float(clean_correct) / total

                    # Calculate l2 distance
                    delta = (adv_inputs - inputs.to(self.device)).view(
                        batch_size, -1
                    )  # nopep8
                    l2_distance.append(
                        torch.norm(delta[~right_idx], p=2, dim=1)
                    )  # nopep8
                    l2 = torch.cat(l2_distance).mean().item()

                    ########################
                    # Calculate Success Rate
                    _, pred = torch.max(outputs.data, 1)
                    _, clean_pred = torch.max(clean_outputs.data, 1)
                    success_total += labels.size(0)
                    right_idx = (clean_pred == labels.to(self.device)) * (pred != labels.to(self.device))
                    success_correct+= right_idx.sum()
                    success_rate = 100 * float(success_correct)/ clean_correct
                    # Calculate F1-robust
                    f1_robust = 100 * correct / (correct + 0.5*(incorrect + clean_incorrect))
                    
                    # Calculate time computation
                    progress = (step + 1) / total_batch * 100
                    end = time.time()
                    elapsed_time = end - start

                    '''if verbose:
                        self._save_print(
                            progress, rob_acc, l2, elapsed_time, end="\r"
                        )  # nopep8'''

            if save_path is not None:
                adv_input_list.append(adv_inputs.detach().cpu())
                label_list.append(labels.detach().cpu())

                adv_input_list_cat = torch.cat(adv_input_list, 0)
                label_list_cat = torch.cat(label_list, 0)

                save_dict = {
                    "adv_inputs": adv_input_list_cat,
                    "labels": label_list_cat,
                }  # nopep8

                if save_predictions:
                    pred_list.append(pred.detach().cpu())
                    pred_list_cat = torch.cat(pred_list, 0)
                    save_dict["preds"] = pred_list_cat

                if save_clean_inputs:
                    input_list.append(inputs.detach().cpu())
                    input_list_cat = torch.cat(input_list, 0)
                    save_dict["clean_inputs"] = input_list_cat

                if self.normalization_used is not None:
                    save_dict["adv_inputs"] = self.inverse_normalize(
                        save_dict["adv_inputs"]
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.inverse_normalize(
                            save_dict["clean_inputs"]
                        )  # nopep8

                if save_type == "int":
                    save_dict["adv_inputs"] = self.to_type(
                        save_dict["adv_inputs"], "int"
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.to_type(
                            save_dict["clean_inputs"], "int"
                        )  # nopep8

                save_dict["save_type"] = save_type
                torch.save(save_dict, save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end="\n")

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time, success_rate, f1_robust


class OTRADES(TRADES):
    r"""
    Adversarial training in 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Attributes:
        self.rmodel : rmodel.
        self.device : device where rmodel is.
        self.optimizer : optimizer.
        self.scheduler : scheduler (Automatically updated).
        self.curr_epoch : current epoch starts from 1 (Automatically updated).
        self.curr_iter : current iters starts from 1 (Automatically updated).

    Arguments:
        rmodel (nn.Module): rmodel to train.
        eps (float): strength of the attack or maximum perturbation.
        alpha (float): step size.
        steps (int): number of steps.
        beta (float): trade-off regularization parameter.
    """

    def __init__(self, rmodel, eps, alpha, steps, beta):
        super().__init__(rmodel,eps,alpha,steps,beta)
        self.atk = TPGD(rmodel, eps, alpha, steps)
        self.beta = beta

    def calculate_cost(self, train_data, reduction="mean"):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits_clean = self.rmodel(images)

        adv_images = self.atk(images)
        logits_adv = self.rmodel(adv_images)
        
        
        
        
        
        L=len(logits_adv)
        
        if method==0:
            #random selection
            indices=torch.randint(L, (int(L/2),))
        elif method==1:
            #IAT UNCERTAINTY SAMPLING
            out, inds = torch.max(logits_adv,dim=1)
            indices=torch.argsort(out.abs(),dim=0)[:int(L/2)].to(self.device)
        elif method==2:
            #IAT MARGIN SAMPLING
            #offset=int(L/(100*2))*int(self.curr_epoch)
            #idx=torch.topk(logits_adv,2,dim=1).indices
            #values=torch.diff(torch.topk(logits_adv,2,dim=1).values)
            #indices=torch.argsort(values.abs(),dim=0)[offset:int(L/2)+offset]
            #indices=torch.argsort(values.abs(),dim=0)[int(L/2)-offset:L-offset]
            #indices=indices.squeeze().to(self.device)
            #IAT MARGIN SAMPLING
            idx=torch.topk(logits_adv,2,dim=1).indices
            values=torch.diff(torch.topk(logits_adv,2,dim=1).values)
            indices=torch.argsort(values.abs(),dim=0)[:int(L/2)]
            indices=indices.squeeze().to(self.device)
        elif method==3:
            #IAT HIGHEST ENTROPY SAMPLING
            offset=int(L/(100*2))*int(self.curr_epoch)
            e=torch.tensor([calc_entropy(a).item() for a in logits_adv])
            #indices=torch.argsort(e,dim=0)[offset:offset+int(L/2)].to(self.device)
            indices=torch.argsort(e,dim=0)[int(L/2)-offset:L-offset].to(self.device)
        elif method==4:
            out, inds = torch.max(logits_adv,dim=1)
            indices_1=torch.argsort(out.abs(),dim=0)[:int(L/2)].to(self.device)
            idx=torch.topk(logits_adv,2,dim=1).indices
            values=torch.diff(torch.topk(logits_adv,2,dim=1).values)
            indices_2=torch.argsort(values.abs(),dim=0)[:int(L/2)]
            indices_2=indices_2.squeeze().to(self.device)
            e=torch.tensor([calc_entropy(a).item() for a in logits_adv])
            indices_3=torch.argsort(e,dim=0)[int(L/2):].to(self.device)
            intersection1=torch.tensor([x.item() for x in indices_1.to(self.device) if x in indices_2.to(self.device)])
            intersection2=torch.tensor([x.item() for x in indices_2.to(self.device) if x in indices_3.to(self.device) and x not in indices_1.to(self.device)])
            intersection3=torch.tensor([x.item() for x in indices_1.to(self.device) if x in indices_3.to(self.device) and x not in indices_2.to(self.device)])
            a=len(intersection1)
            b=len(intersection2)
            c=len(intersection3)
            indices=torch.cat( (intersection1,intersection2,intersection3) ).to(self.device)
            if int(L/2)<len(indices):
              indices=indices[:int(L/2)]
            else:
              M=int(( int(L/2)-len(indices) )/3)
              indices=torch.cat((indices,indices_1[:M],indices_2[M:],indices_3[:M]),0).long().to(self.device)

        logits_adv=logits_adv[indices]
        labels=labels[indices]
        logits_clean=logits_clean[indices]
        
        
        
        
        
        
        loss_ce = nn.CrossEntropyLoss(reduction=reduction)(logits_clean, labels)
        probs_clean = F.softmax(logits_clean, dim=1)
        log_probs_adv = F.log_softmax(logits_adv, dim=1)
        loss_kl = nn.KLDivLoss(reduction="none")(log_probs_adv, probs_clean).sum(dim=1)

        cost = loss_ce + self.beta * loss_kl

        self.add_record_item("Loss", cost.mean().item())
        self.add_record_item("CELoss", loss_ce.mean().item())
        self.add_record_item("KLLoss", loss_kl.mean().item())

        return cost.mean() if reduction == "mean" else cost
        
    def record_during_eval(self):
        for flag, loader in self.rob_dict["loaders"].items():
            self.dict_record["Clean" + flag] = self.rmodel.eval_accuracy(loader)

            eps = self.rob_dict.get("eps")
            if eps is not None:
                atk = SIFGSM(rmodel, eps=eps)
                self.dict_record["FGSM" + flag] = rmodel.eval_rob_accuracy(loader, atk)
                self.dict_record["FF1" + flag] = atk.save(loader,return_verbose=True)[4]
                '''self.dict_record["FGSM" + flag] = self.rmodel.eval_rob_accuracy_fgsm(
                    loader, eps=eps, verbose=False
                )'''
                steps = self.rob_dict.get("steps")
                alpha = self.rob_dict.get("alpha")
                if steps is not None:
                    #self.dict_record["PGD" + flag] = self.rmodel.eval_rob_accuracy_pgd(loader, eps=eps, alpha=alpha, steps=steps, verbose=False)
                    atk = SIPGD(rmodel, eps=eps, alpha=alpha, steps=steps, random_start=True)
                    self.dict_record["PGD" + flag] = rmodel.eval_rob_accuracy(loader, atk)
                    self.dict_record["F1" + flag] = atk.save(loader,return_verbose=True)[4]
                '''set_of_steps=[10,20,30,40,50,60,70,80,90,100]
                for steps in set_of_steps:
                  self.dict_record["PGD{}".format(steps) + flag] = self.rmodel.eval_rob_accuracy_pgd(loader, eps=eps, alpha=alpha, steps=steps, verbose=False)'''
            
            device = next(rmodel.parameters()).device
            atk = SIAutoAttack(rmodel,norm="Linf", eps=eps, version="standard", n_classes=N_CLASSES)
            self.dict_record["AA" + flag] = rmodel.eval_rob_accuracy(loader, atk)
            self.dict_record["F1A" + flag] = atk.save(loader,return_verbose=True)[4]
            #self.dict_record["AA" + flag] = self.rmodel.eval_rob_accuracy_autoattack(loader,eps, version="standard", norm="Linf")
            
            '''
            atk = SIDeepFool(rmodel, steps=50, overshoot=0.02)
            self.dict_record["DF" + flag] = rmodel.eval_rob_accuracy(loader, atk)
            
            self.dict_record["SRD" + flag] = atk.save(loader,return_verbose=True)[3]
            self.dict_record["F1D" + flag] = atk.save(loader,return_verbose=True)[4]
            '''

            std = self.rob_dict.get("std")
            if std is not None:
                self.dict_record["GN" + flag] = self.rmodel.eval_rob_accuracy_gn(
                    loader, std=std, verbose=False
                )

FROZEN_MODEL_PATH= F"fm10.pt"

PATH = "./models/"
NAME = "Sample"
SAVE_PATH = PATH + NAME
MODEL_NAME = "ResNet18"
DATA = "CIFAR10"
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]
N_VALIDATION = 1000
N_CLASSES = 10
EPOCH = 200
EPS = 8/255
ALPHA = 2/255
STEPS = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


if len(sys.argv) < 2:
    print("Please provide method of choice:")
    print("0) Standard AT (Random Acquisition)")
    print("1) Informative AT (Most Uncertain Acquisition)")
    print("2) Informative AT (Least Margin Acquisition)")
    print("3) Informative AT (Highest Entropy Acquisition)")
    print("4) Informative AT (Union Acquisition)")
    exit()
method = int(sys.argv[1])
if method >=0 and method<=4:
    print("You selected {}".format(method))
    print("0) Standard AT (Random Acquisition)")
    print("1) Informative AT (Most Uncertain Acquisition)")
    print("2) Informative AT (Least Margin Acquisition)")
    print("3) Informative AT (Highest Entropy Acquisition)")
    print("4) Informative AT (Union Acquisition)")
else:
    print("Choice must be in range [0,4]")
    exit()

setting= int(sys.argv[2])
if setting >=0 and setting<=2:
    print("You Selected {}".format(setting))
    print("0) WideResNet-34-10 for CIFAR-10")
    print("1) WideResNet-34-10 for CIFAR-100")
    print("2) PreAct ResNet for SVHN")
else:
    print("Choice must be in range[0,2]")
    exit()

model_choice=int(sys.argv[3])

if setting==0:
    train_data = dsets.CIFAR10(root='./data',
                           train=True,
                           download=True,
                           transform=transform)
    test_data  = dsets.CIFAR10(root='./data',
                           train=False,
                           download=True,
                           transform=transform)
elif setting==1:
    train_data = dsets.CIFAR100(root='./data',
                           train=True,
                           download=True,
                           transform=transform)
    test_data  = dsets.CIFAR100(root='./data',
                           train=False,
                           download=True,
                           transform=transform)
    '''id_dict = {}
    for i, line in enumerate(open('tiny-imagenet-200/wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i

    transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
    train_data = TrainTinyImageNetDataset(id=id_dict, transform = transform)
    test_data = TestTinyImageNetDataset(id=id_dict, transform=transform)'''
elif setting==2:
    train_data = dsets.SVHN(root='./data',
                           split='train',
                           download=True,
                           transform=transform)
    test_data  = dsets.SVHN(root='./data',
                           split='test',
                           download=True,
                           transform=transform)

if setting==0:
    MODEL_NAME='ResNet18'
    batch_size = 128
elif setting==1:
    MODEL_NAME='WRN34-10'
    batch_size = 256
    N_CLASSES=200
elif setting==2:
    MODEL_NAME='ResNet18'
    batch_size = 128

if model_choice==0:
    MODEL_NAME='ResNet18'
elif model_choice==1:
    MODEL_NAME='WRN34-10'

train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_data,
                         batch_size=batch_size,
                         shuffle=False)

model = mair.utils.load_model(model_name=MODEL_NAME,
                              n_classes=N_CLASSES).cuda() # Load model
rmodel = mair.RobModel(model, n_classes=N_CLASSES,
                       normalization_used={'mean':MEAN, 'std':STD}).cuda()
rmodel.load_state_dict(torch.load(FROZEN_MODEL_PATH, weights_only=True))
rmodel.eval()        
        
trainer = OTRADES(rmodel, eps=EPS, alpha=ALPHA, steps=STEPS,beta=10)
trainer.record_rob(train_loader, test_loader, eps=EPS, alpha=2/255, steps=10, std=0.1,
                   n_train_limit=N_VALIDATION, n_val_limit=N_VALIDATION)
trainer.setup(optimizer="SGD(lr=0.1, momentum=0.9, weight_decay=0.0005)",
              scheduler="Step(milestones=[100, 150], gamma=0.1)",
              scheduler_type="Epoch",
              minimizer=None,
              n_epochs=100, n_iters=len(train_loader)
             )
trainer.fit(train_loader=train_loader, n_epochs=100,
            save_path=SAVE_PATH, save_best={"Clean(Val)":"HBO", "PGD(Val)":"HB"},
            save_type=None, save_overwrite=True, record_type="Epoch")
