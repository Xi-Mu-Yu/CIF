from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

import logging
import os
import pandas as pd


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _build_model(self):
        # model input depends on data
        # train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = test_data.max_seq_len  # redefine seq_len      在这里改了输入序列长
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.enc_in = test_data.X.shape[2]  # redefine enc_in    在这里改了
        self.args.num_class = len(np.unique(test_data.y))
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.swa:
                    outputs = self.swa_model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        # print(trues_onehot.shape)
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict



    # 在train方法外部设置logging


    # 在train方法中使用 logger
    def train(self, setting,logger):
        # 设置日志文件路径
        # log_file = './train_log.txt'
        # logger = setup_logger(log_file)
        # print('11111111111111111111111111111111111111111111')
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        # print('11111111111111111111111111111111111111111111')

    

        path = (
                "./checkpoints/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
                + setting
                + "/"
        )

        # 构建日志文件路径
        # log_dir = './log/' + self.args.task_name + "/" + self.args.model_id + "/" + self.args.model+'/'+ setting + "/"
        #
        # # 创建日志目录（如果不存在）
        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir)
        #
        # # 最终的日志文件路径
        # log_file = os.path.join(log_dir, 'log.txt')
        #
        # # 现在你可以将日志文件传递给 logger 或其他操作
        #
        # logger = setup_logger(log_file)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        logger.info("train_steps: %d", train_steps)

        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5,logger=logger
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Reset seed-level SNR stats once per training seed/run.
        model_unwrapped = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_unwrapped, "reset_snr_seed_stats"):
            model_unwrapped.reset_snr_seed_stats()

        # Configure per-iter txt logging + terminal print interval.
        if getattr(model_unwrapped, "snr_enabled", False):
            # Save all iter stats to txt; print only every 50 iter to Terminal.
            model_unwrapped.snr_log_interval = 50
            snr_txt_path = os.path.join(path, f"snr_iter_stats_seed{self.args.seed}.txt")
            if hasattr(model_unwrapped, "set_snr_iter_txt_path"):
                model_unwrapped.set_snr_iter_txt_path(snr_txt_path)

        # # Helper to unwrap DataParallel wrappers
        # def _unwrap(m):
        #     return m.module if hasattr(m, "module") else m

        # # Record learnable a/b per seed (within this training run)
        # ab_records = []
        # model_for_ab = _unwrap(self.model)
        # has_learnab = (
        #     hasattr(model_for_ab, "a")
        #     and hasattr(model_for_ab, "b")
        #     and isinstance(model_for_ab.a, torch.nn.Parameter)
        # )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            # Reset SNR accumulators at epoch start (if model supports it)
            model_unwrapped = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(model_unwrapped, "reset_snr_epoch_stats"):
                model_unwrapped.reset_snr_epoch_stats()

            logger.info('')  # 记录一个空行
            logger.info('')  # 记录一个空行
            epoch_time = time.time()
            

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                padding_mask = padding_mask.float().to(self.device)

                label = label.to(self.device)


                outputs = self.model(batch_x, padding_mask, None, None)


                loss = criterion(outputs, label.long())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logger.info(
                        "\titers: %d, epoch: %d | loss: %.7f",
                        i + 1, epoch + 1, loss.item()
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info("\tspeed: %.4fs/iter; left time: %.4fs", speed, left_time)
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

                # If a/b are learnable, record and print every training iteration.
                # if has_learnab:
                #     model_for_ab = _unwrap(self.model)
                #     global_step = epoch * train_steps + (i + 1)
                #     a_val = float(model_for_ab.a.detach().cpu().item())
                #     b_val = float(model_for_ab.b.detach().cpu().item())
                #     ab_records.append(
                #         {
                #             "global_step": global_step,
                #             "epoch": epoch + 1,
                #             "iter": i + 1,
                #             "a": a_val,
                #             "b": b_val,
                #         }
                #     )
                    # logger.info(
                    #     f"[learnab] seed={self.args.seed} global_step={global_step} "
                    #     f"epoch={epoch+1} iter={i+1} a={a_val:.6f} b={b_val:.6f}"
                    # )

            self.swa_model.update_parameters(self.model)

            logger.info("Epoch: %d cost time: %.4f", epoch + 1, time.time() - epoch_time)
            train_loss = np.average(train_loss)
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

            logger.info(
                "Epoch: %d, Steps: %d | Train Loss: %.5f\n"
                "Validation results --- Loss: %.5f, Accuracy: %.5f, Precision: %.5f, Recall: %.5f, F1: %.5f, AUROC: %.5f, AUPRC: %.5f\n"
                "Test results --- Loss: %.5f, Accuracy: %.5f, Precision: %.5f, Recall: %.5f, F1: %.5f, AUROC: %.5f, AUPRC: %.5f",
                epoch + 1, train_steps, train_loss, vali_loss,
                val_metrics_dict['Accuracy'], val_metrics_dict['Precision'], val_metrics_dict['Recall'],
                val_metrics_dict['F1'], val_metrics_dict['AUROC'], val_metrics_dict['AUPRC'],
                test_loss, test_metrics_dict['Accuracy'], test_metrics_dict['Precision'],
                test_metrics_dict['Recall'], test_metrics_dict['F1'], test_metrics_dict['AUROC'],
                test_metrics_dict['AUPRC']
            )

            # Print epoch-level SNR summary (if enabled)
            if hasattr(model_unwrapped, "get_snr_epoch_stats"):
                snr_stats = model_unwrapped.get_snr_epoch_stats()
                if snr_stats is not None:
                    logger.info(
                        "[SNR-epoch] trial Δ=%.6f±%.6f dB (improved_ratio=%.3f) | "
                        "band Δ=%.6f±%.6f dB (improved_ratio=%.3f) | updates=%d",
                        snr_stats["trial_delta_mean"],
                        snr_stats["trial_delta_std"],
                        snr_stats["trial_improved_ratio"],
                        snr_stats["band_delta_mean"],
                        snr_stats["band_delta_std"],
                        snr_stats["band_improved_ratio"],
                        snr_stats["n_updates"],
                    )
                    # Also report which changed channels improved (top/bottom 3 by mean Δ).
                    if "trial_top3" in snr_stats and "trial_bottom3" in snr_stats:
                        logger.info(
                            "[SNR-epoch-ch] trial_top3=%s trial_bottom3=%s | band_top3=%s band_bottom3=%s",
                            ",".join(snr_stats.get("trial_top3", [])),
                            ",".join(snr_stats.get("trial_bottom3", [])),
                            ",".join(snr_stats.get("band_top3", [])),
                            ",".join(snr_stats.get("band_bottom3", [])),
                        )
                    # Positive/negative contributions and median on changed channels.
                    if "trial_pos_mean_ch" in snr_stats and "band_pos_mean_ch" in snr_stats:
                        logger.info(
                            "[SNR-epoch-ch-stats] trial pos-mean=%.6f neg-mean=%.6f median=%.6f | band pos-mean=%.6f neg-mean=%.6f median=%.6f",
                            snr_stats.get("trial_pos_mean_ch", float("nan")),
                            snr_stats.get("trial_neg_mean_ch", float("nan")),
                            snr_stats.get("trial_median_ch", float("nan")),
                            snr_stats.get("band_pos_mean_ch", float("nan")),
                            snr_stats.get("band_neg_mean_ch", float("nan")),
                            snr_stats.get("band_median_ch", float("nan")),
                        )
                    if "trial_delta_all" in snr_stats and len(snr_stats.get("trial_delta_all", [])) > 0:
                        # Only print full list when changed-channels is not too large.
                        if len(snr_stats.get("trial_delta_all", [])) <= 16:
                            logger.info(
                                "[SNR-epoch-ch-list] trial_delta_all=%s | band_delta_all=%s",
                                ",".join(snr_stats.get("trial_delta_all", [])),
                                ",".join(snr_stats.get("band_delta_all", [])),
                            )

            early_stopping(
                -val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        best_model_path = path + "checkpoint.pth"
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        # Print per-seed "how many channels improved" for trial-avg SNR.
        model_unwrapped = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_unwrapped, "get_snr_seed_stats"):
            snr_seed_stats = model_unwrapped.get_snr_seed_stats()
            if snr_seed_stats is not None:
                logger.info(
                    "[SNR-seed] improved_count=%d/%d (ratio=%.4f) | improved_global_idx=%s | mean_delta_improved=%s",
                    snr_seed_stats.get("improved_count", -1),
                    snr_seed_stats.get("n_changed_channels", -1),
                    snr_seed_stats.get("improved_ratio", float("nan")),
                    snr_seed_stats.get("improved_global_idx", []),
                    snr_seed_stats.get("improved_mean_delta_ch", []),
                )

        # Save per-seed a/b trajectory to Excel
        # if has_learnab and len(ab_records) > 0:
        #     df_ab = pd.DataFrame(ab_records)
        #     a_mean = float(df_ab["a"].mean())
        #     a_var = float(df_ab["a"].var(ddof=0))
        #     b_mean = float(df_ab["b"].mean())
        #     b_var = float(df_ab["b"].var(ddof=0))

        #     summary = pd.DataFrame(
        #         [
        #             {
        #                 "seed": self.args.seed,
        #                 "a_mean": a_mean,
        #                 "a_var": a_var,
        #                 "b_mean": b_mean,
        #                 "b_var": b_var,
        #                 "n_points": len(ab_records),
        #             }
        #         ]
        #     )

        #     out_dir = os.path.join("./results", "learnab_ab_excel")
        #     os.makedirs(out_dir, exist_ok=True)
        #     excel_path = os.path.join(
        #         out_dir,
        #         f"{self.args.task_name}_{self.args.data}_{self.args.model_id}_{self.args.model}_seed{self.args.seed}.xlsx",
        #     )

        #     with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        #         df_ab.to_excel(writer, sheet_name="trajectory", index=False)
        #         summary.to_excel(writer, sheet_name="summary", index=False)

        #     logger.info(f"[learnab] Saved a/b Excel to: {excel_path}")

        return self.model

    def test(self, setting, test=0, logger=None):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        if test:
            # 使用 logger 或 print
            if logger is not None:
                logger.info("loading model")
            else:
                print("loading model")

            path = (
                "./checkpoints/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # 保存结果
        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 使用 logger 或 print 输出验证和测试结果
        result_message = (
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )

        if logger is not None:
            logger.info(result_message)
        else:
            print(result_message)

        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return
