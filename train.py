import os

import matplotlib

import matplotlib.pyplot as plt

from read_dataset import get_dataset
from tfnet import TFNet
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
from config import Config
import utils

LVL_ERROR = 10
LVL_INFO = 5
LVL_DEBUG = 1

LOG = 1  # Will log all mesages with lvl greater than this
SAVE_LOG = True


class Train:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _log(self, message, lvl=LVL_INFO):
        n_msg = f"{self.run_name}{message}"
        if lvl >= LOG:
            print(n_msg)

    def train(self):

        self._set_rets_path()
        self._create_results_dirs()
        self.print_run_params()

        device = self._get_device()
        model = self._get_model().to(device)
        optimizer = self._get_optimizer(model)
        loss_seg, loss_dec = self.get_loss_weights(True), self.get_loss_weights(False)

        train_loader = get_dataset("train", self.cfg)
        test_loader = get_dataset("test", self.cfg)

        tensorboard_writer = SummaryWriter(log_dir=self.cfg.tensorboard_dir) if self.cfg.WRITE_TENSORBOARD else None

        train_rets = self._train_model(device, model, train_loader, loss_seg, loss_dec, optimizer, test_loader,
                                       tensorboard_writer)
        self._save_train_ret(train_rets)
        self._save_model(model)

        self.eval(model, device, self.cfg.sava_imgs, False, False, self.outputs_path)
        self._save_params()

    def _get_device(self):
        return f"cuda:{self.cfg.gpu}"

    def _get_model(self):
        tfnet = TFNet(self._get_device(), self.cfg.input_w, self.cfg.input_h, self.cfg.input_c)
        return tfnet

    def _get_optimizer(self, model):
        # return torch.optim.SGD([
        #     {"params": model.patten.parameters(), "lr": self.cfg.learn_rate_seg}
        # ],
        #     lr=self.cfg.learn_rate)
        return torch.optim.SGD(model.parameters(), self.cfg.LEARNING_RATE)


    def _train_model(self, device, model, train_loader, criterion_seg, criterion_dec, optimizer, test_loader,
                     tensorboard_writer):
        losses = []
        validation_data = []
        max_validation = -1
        validation_step = self.cfg.test_frequency

        num_epochs = self.cfg.epochs
        samples_per_epoch = len(train_loader) * self.cfg.batch_size

        self.set_dec_gradient_multiplier(model, 0.0)

        for epoch in range(num_epochs):
            if epoch % self.cfg.save_frequency == 0:
                self._save_model(model, f"ep_{epoch:02}.pth")

            model.train()

            weight_loss_seg, weight_loss_dec = self.get_loss_weights(epoch)
            dec_gradient_multiplier = self.get_dec_gradient_multipliter()
            self.set_dec_gradient_multiplier(model, dec_gradient_multiplier)

            epoch_loss_seg, epoch_loss_dec, epoch_loss = 0, 0, 0
            epoch_correct = 0

            from timeit import default_timer as timer

            time_acc = 0
            start = timer()
            for iter_index, (data) in enumerate(train_loader):
                start_1 = timer()
                curr_loss_seg, curr_loss_dec, curr_loss, correct = self.training_iteration(data, device, model,
                                                                                           criterion_seg,
                                                                                           criterion_dec,
                                                                                           optimizer, weight_loss_seg,
                                                                                           weight_loss_dec,
                                                                                           tensorboard_writer, (
                                                                                                       epoch * samples_per_epoch + iter_index))
                end_1 = timer()
                time_acc = time_acc + (end_1 - start_1)

                epoch_loss_seg += curr_loss_seg
                epoch_loss_dec += curr_loss_dec
                epoch_loss += curr_loss

                epoch_correct += correct
            end = timer()

            epoch_loss_seg = epoch_loss_seg / samples_per_epoch
            epoch_loss_dec = epoch_loss_dec / samples_per_epoch
            epoch_loss = epoch_loss / samples_per_epoch
            losses.append((epoch_loss_seg, epoch_loss_dec, epoch_loss, epoch))

            self._log(
                f"Epoch {epoch + 1}/{num_epochs} ==> avg_loss_seg={epoch_loss_seg:.5f}, avg_loss_dec={epoch_loss_dec:.5f}, avg_loss={epoch_loss:.5f}, correct={epoch_correct}/{samples_per_epoch}, in {end - start:.2f}s/epoch (fwd/bck in {time_acc:.2f}s/epoch)")

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss/Train/segmentation", epoch_loss_seg, epoch)
                tensorboard_writer.add_scalar("Loss/Train/classification", epoch_loss_dec, epoch)
                tensorboard_writer.add_scalar("Loss/Train/joined", epoch_loss, epoch)
                tensorboard_writer.add_scalar("Accuracy/Train/", epoch_correct / samples_per_epoch, epoch)

            if self.cfg.test_during_train and (epoch % validation_step == 0 or epoch == num_epochs - 1):
                validation_ap, validation_accuracy = self.eval_model(model, device, test_loader, False, False, True,
                                                                     None)
                validation_data.append((validation_ap, epoch))

                if validation_ap > max_validation:
                    max_validation = validation_ap
                    self._save_model(model, "best.pth")

                model.train()
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("Accuracy/Validation/", validation_accuracy, epoch)
        return losses, validation_data

    def _save_train_ret(self, train_rets):
        losses, validation_data = train_rets
        ls, ld, l, le = map(list, zip(*losses))
        plt.plot(le, l, label="Loss", color="red")
        plt.plot(le, ls, label="Loss seg")
        plt.plot(le, ld, label="Loss dec")
        plt.ylim(bottom=0)
        plt.grid()
        plt.xlabel('Epochs')
        if self.cfg.test_during_train:
            v, ve = map(list, zip(*validation_data))
            plt.twinx()
            plt.plot(ve, v, label="Validation AP", color="Green")
            plt.ylim((0, 1))
        plt.legend()
        plt.savefig(os.path.join(self.run_path, "loss_val"), dpi=200)

        df_loss = pd.DataFrame(data={"loss_seg": ls, "loss_dec": ld, "loss": l, "epoch": le})
        df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

        if self.cfg.test_during_train:
            df_loss = pd.DataFrame(data={"validation_data": ls, "loss_dec": ld, "loss": l, "epoch": le})
            df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

    def _save_model(self, model, name="final.pth"):
        output_name = os.path.join(self.model_path, name)
        self._log(f"saving current model to {output_name}")
        if os.path.exists(output_name):
            os.remove(output_name)
        torch.save(model.state_dict(), output_name)

    def eval(self, model, device, save_images, plot_Seg, is_validation, save_folder):
        self.reload_model(model)
        test_loader = get_dataset("TEST", self.cfg)
        self.eval_model(model, device, test_loader, save_images, plot_Seg, is_validation, save_folder)

    def eval_model(self, model, device, test_loader, save_images, plot_Seg, is_validation, save_folder):
        model.eval()

        dsize = self.cfg.input_w, self.cfg.input_h

        res = []
        predictions, gts = [], []

        for data in test_loader:
            image, seg_mask, seg_loss_mask, _, sample_name = data
            image, seg_mask = image.to(device), seg_mask.to(device)
            is_pos = (seg_mask.max() > 0).reshape((1, 1)).to(device).item()
            prediction, pre_seg = model(image)
            pre_seg = nn.Sigmoid()(pre_seg)
            prediction = nn.Sigmoid()(prediction)

            prediction = prediction.item()
            image = image.detach().cpu().numpy()
            pre_seg = pre_seg.detach().cpu().numpy()
            seg_mask = seg_mask.detach().cpu().numpy()

            predictions.append(prediction)
            gts.append(is_pos)
            res.append((prediction, None, None, is_pos, sample_name[0]))
            if not is_validation:
                if save_images:
                    image = cv2.resize(np.transpose(image[0, :, :, :], (1, 2, 0)), dsize)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    pre_seg = cv2.resize(pre_seg[0, 0, :, :], dsize) if len(pre_seg.shape) == 4 else cv2.resize(
                        pre_seg[0, :, :], dsize)
                    seg_mask = cv2.resize(seg_mask[0, 0, :, :], dsize)
                    if self.cfg.is_seg_loss_weighted:
                        seg_loss_mask = cv2.resize(seg_loss_mask.numpy()[0, 0, :, :], dsize)
                        utils.plot_sample(sample_name[0], image, pre_seg, seg_loss_mask, save_folder,
                                          decision=prediction, plot_seg=plot_Seg)
                    else:
                        utils.plot_sample(sample_name[0], image, pre_seg, seg_mask, save_folder, decision=prediction,
                                          plot_seg=plot_Seg)
            if is_validation:
                metrics = utils.get_metrics(np.array(gts), np.array(predictions))
                FP, FN, TP, TN = list(map(sum, [metrics["FP"], metrics["FN"], metrics["TP"], metrics["TN"]]))
                self._log(
                    f"VALIDATION || AUC={metrics['AUC']:f}, and AP={metrics['AP']:f}, with best thr={metrics['best_thr']:f} "
                    f"at f-measure={metrics['best_f_measure']:.3f} and FP={FP:d}, FN={FN:d}, TOTAL SAMPLES={FP + FN + TP + TN:d}")

                return metrics["AP"], metrics["accuracy"]
            else:
                utils.evaluate_metrics(res, self.run_path, self.run_name)

    def _save_params(self):
        params = self.cfg.get_as_dict()
        params_lines = sorted(map(lambda e: e[0] + ':' + str(e[1]) + '\n', params.items()))
        fname = os.path.join(self.run_path, "run.params.txt")
        with open(fname, "w+") as f:
            f.writelines(params_lines)

    def _set_rets_path(self):
        self.run_name = self.cfg.dataset
        ret_path = os.path.join(self.cfg.ret_dir, self.cfg.dataset)
        self.tensorboard_path = os.path.join(ret_path, "tensorboard", self.run_name)
        run_path = os.path.join(ret_path, self.run_name)

        self.run_path = run_path
        self.model_path = os.path.join(run_path, "models")
        self.outputs_path = os.path.join(run_path, "test_outputs")

    def _create_results_dirs(self):
        list(map(utils.create_folder, [self.run_path, self.model_path, self.outputs_path, ]))

    def set_dec_gradient_multiplier(self, model, multiplier):
        model.set_gradient_multipliers(multiplier)

    def get_dec_gradient_multipliter(self):
        if self.cfg.GRADIENT_ADJUSTMENT:
            grad_m = 0
        else:
            grad_m = 1
        self._log(f"Returning dec_gradient_multiplier {grad_m}", LVL_DEBUG)
        return grad_m

    def get_loss_weights(self, epoch):
        total_epochs = float(self.cfg.epochs)
        if self.cfg.dyn_balance_loss:
            seg_loss_weight = 1 - (epoch / total_epochs)
            dec_loss_weight = self.cfg.delta_cls_loss * epoch / total_epochs
        else:
            seg_loss_weight = 1
            dec_loss_weight = self.cfg.delta_cls_loss
        self._log(f"seg_loss_weight {seg_loss_weight}  dec_loss_weight {dec_loss_weight}")
        return seg_loss_weight, dec_loss_weight

    def reload_model(self, model):
        path = os.path.join(self.model_path, "best.pth")
        self._log("正在加载最好的模型")
        model.load_state_dict(torch.load(path))

    def training_iteration(self, data, device, model, criterion_seg, criterion_dec, optimizer, weight_loss_seg,
                           weight_loss_dec, tensorboard_writer, iter_idx):
        global loss_seg
        images, seg_masks, seg_loss_masks, is_segmented, _ = data

        batch_size = self.cfg.batch_size

        num_subiters = batch_size

        total_loss = 0
        total_correct = 0

        optimizer.zero_grad()

        total_loss_seg = 0
        total_loss_dec = 0

        for sub_iter in range(num_subiters):
            images_ = images[sub_iter * 1:(sub_iter + 1) * 1, :, :, :].to(device)
            seg_masks_ = seg_masks[sub_iter * 1:(sub_iter + 1) * 1, :, :, :].to(device)
            seg_loss_masks_ = seg_loss_masks[sub_iter * 1:(sub_iter + 1) * 1, :, :, :].to(device)
            is_pos_ = seg_masks_.max().reshape((1, 1)).to(device)

            if tensorboard_writer is not None and iter_idx % 100 == 0:
                tensorboard_writer.add_image(f"{iter_idx}/image", images_[0, :, :, :])
                tensorboard_writer.add_image(f"{iter_idx}/seg_mask", seg_masks[0, :, :, :])
                tensorboard_writer.add_image(f"{iter_idx}/seg_loss_mask", seg_loss_masks_[0, :, :, :])

            decision, output_seg_mask = model(images_)

            if is_segmented[sub_iter]:
                if self.cfg.is_seg_loss_weighted:
                    loss_seg = torch.mean(criterion_seg(output_seg_mask, seg_masks_) * seg_loss_masks_)
                else:
                    loss_seg = criterion_seg(output_seg_mask, seg_masks_)

                loss_dec = criterion_dec(decision, is_pos_)
                total_loss_seg += loss_seg.item()
                total_loss_dec += loss_dec.item()

                total_correct += (decision > 0.0).item() == is_pos_.item()
                loss = weight_loss_dec * loss_dec + weight_loss_seg * loss_seg
            else:
                loss_dec = criterion_dec(decision, is_pos_)
                total_loss_dec += loss_dec.item()

                total_correct += (decision > 0.0).item() == is_pos_.item()
                loss = weight_loss_dec * loss_dec

            total_loss += loss.item()
            # loss.backward()
            loss_dec.backward()
            loss_seg.backward()
        optimizer.step()
        optimizer.zero_grad()

        return total_loss_seg, total_loss_dec, total_loss, total_correct


    def print_run_params(self):
        for l in sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", self.cfg.get_as_dict().items())):
            k, v = l.split(":")
            self._log(f"{k:25s} : {str(v.strip())}")