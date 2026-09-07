import argparse
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np
import logging
import os



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# def setup_logger(log_file):
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     # 创建日志格式
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     # 创建文件处理器
#     file_handler = logging.FileHandler(log_file)
#     file_handler.setFormatter(formatter)
#     # 创建控制台处理器
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(formatter)
#     # 将文件和控制台处理器添加到logger
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)
#     return logger

# def setup_logger(log_file):
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)

#     # 清除之前的处理器，避免重复添加
#     for handler in logger.handlers[:]:
#         logger.removeHandler(handler)
#         handler.close()

#     # 创建日志格式
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

#     # 创建文件处理器
#     file_handler = logging.FileHandler(log_file)
#     file_handler.setFormatter(formatter)

#     # 创建控制台处理器
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(formatter)

#     # 将文件和控制台处理器添加到logger
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

#     return logger

import logging

def setup_logger(log_file, mode='w'):
    """
    设置日志记录器。

    :param log_file: 日志文件路径
    :param mode: 文件模式，'w' 表示覆盖，'a' 表示追加
    :return: 配置好的 logger 对象
    """
    logger = logging.getLogger()
    
    # 清除之前的处理器，避免重复添加
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setFormatter(formatter)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 将文件和控制台处理器添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    """fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)"""

    parser = argparse.ArgumentParser(description="TimesNet")

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        default="long_term_forecast",
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
    )
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Autoformer",
        help="model name, options: [Autoformer, Transformer, TimesNet]",
    )

    # data loader
    parser.add_argument(
        "--data", type=str, required=True, default="ETTm1", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    # parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
    )
    parser.add_argument(
        "--inverse", action="store_true", help="inverse output data", default=False
    )

    # inputation task
    parser.add_argument("--mask_rate", type=float, default=0.25, help="mask ratio")

    # anomaly detection task
    parser.add_argument(
        "--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%)"
    )

    # model define for baselines
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )
    parser.add_argument(
        "--no_inter_attn",
        action="store_true",
        help="whether to use inter-attention in encoder, using this argument means not using inter-attention",
        default=False,
    )
    parser.add_argument(
        "--chunk_size", type=int, default=16, help="chunk_size used in LightTS"
    )
    parser.add_argument(
        "--patch_len", type=int, default=16, help="patch_len used in PatchTST"
    )
    parser.add_argument("--stride", type=int, default=8, help="stride used in PatchTST")
    parser.add_argument(
        "--sampling_rate", type=int, default=256, help="frequency sampling rate"
    )
    parser.add_argument(
        "--patch_len_list",
        type=str,
        default="2,4,8",
        help="a list of patch len used in Medformer",
    )
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="whether to use single channel patching for Medformer",
        default=False,
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        default="flip,shuffle,frequency,jitter,mask,drop",
        help="A comma-seperated list of augmentation types (none, jitter or scale). "
             "Randomly applied to each granularity. "
             "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",
    )

    # 添加不同的卷积核心
    parser.add_argument(
        "--patch_H",
        type=str,
        default="2,4,8",
        help="a list of patch len used in Medformer",
    )

    # optimization
    # parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument(
        "--num_workers", type=int, default=0, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )
    parser.add_argument(
        "--swa",
        action="store_true",
        help="use stochastic weight averaging",
        default=False,
    )

    parser.add_argument("--t", type=int, default=1, help="")
    parser.add_argument("--n", type=int, default=8, help="")
    parser.add_argument("--a", type=float, default=1, help="")
    parser.add_argument("--b", type=float, default=1, help="")
    parser.add_argument('--learnab', type=str2bool, default=True, help='Enable learnab')
    parser.add_argument('--use_cif', type=str2bool, default=False, help='Enable CIF channel fusion')
    parser.add_argument(
        '--cif_select',
        type=str2bool,
        default=False,
        help='Apply CIF only on selected changed channels (subset of the n front/back channels)',
    )
    parser.add_argument(
        '--cif_channel_mask',
        type=int,
        nargs='*',
        default=None,
        help='Local indices (0..n-1) within changed channels where CIF fusion is applied',
    )
    parser.add_argument(
        '--cif_snr_gain_csv',
        type=str,
        default=None,
        help='Trail_snr *_per_channel.csv; auto-set cif_channel_mask to channels with mean snr_delta_ch_db > cif_snr_min_delta',
    )
    parser.add_argument(
        '--cif_snr_splits',
        type=str,
        nargs='*',
        default=['TRAIN'],
        help='Splits used when deriving cif_channel_mask from --cif_snr_gain_csv (default: TRAIN only)',
    )
    parser.add_argument(
        '--cif_snr_min_delta',
        type=float,
        default=0.0,
        help='Minimum mean per-channel SNR delta (dB) to include a channel in cif_select',
    )
    parser.add_argument(
        "--up_dim_list",
        type=str,
        default="19",
        help="Up-dimension list for ADformer spatial block",
    )
    parser.add_argument(
        "--no_temporal_block",
        action="store_true",
        help="Disable ADformer temporal block",
        default=False,
    )
    parser.add_argument(
        "--no_spatial_block",
        action="store_true",
        help="Disable ADformer spatial block",
        default=False,
    )
    parser.add_argument(
        "--ecg_kernel_size",
        type=int,
        default=40,
        help="Kernel size for Inception1d backbone",
    )

    # SNR logging / evaluation
    parser.add_argument(
        "--log_snr",
        type=str2bool,
        default=False,
        help="Enable SNR evaluation inside models (for debugging / analysis)",
    )
    parser.add_argument(
        "--snr_log_interval",
        type=int,
        default=100,
        help="Unused by epoch-level summary; kept for backward compatibility",
    )
    parser.add_argument(
        "--denoise_method",
        type=str,
        default="fusion",
        help="Denoise strategy: fusion | bandpass_notch | wavelet_db4 | median | savgol | none",
    )
    parser.add_argument("--denoise_bp_low", type=float, default=1.0, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--denoise_bp_high", type=float, default=40.0, help="Bandpass high cutoff (Hz)")
    parser.add_argument("--denoise_line_freq", type=float, default=50.0, help="Line noise frequency for notch (Hz)")
    parser.add_argument("--denoise_notch_bw", type=float, default=1.0, help="Half notch width (Hz)")
    parser.add_argument(
        "--denoise_notch_harmonics",
        type=int,
        default=1,
        help="How many harmonics to notch (1 means only line freq)",
    )
    parser.add_argument("--denoise_median_kernel", type=int, default=5, help="Median filter kernel size (odd)")
    parser.add_argument("--denoise_savgol_window", type=int, default=11, help="Savitzky-Golay window length (odd)")
    parser.add_argument("--denoise_savgol_polyorder", type=int, default=2, help="Savitzky-Golay polynomial order")
    parser.add_argument("--denoise_wavelet", type=str, default="db4", help="Wavelet basis for wavelet denoise")
    parser.add_argument("--denoise_wavelet_level", type=int, default=3, help="Wavelet decomposition level")
    parser.add_argument(
        "--denoise_csp_keep_ratio",
        type=float,
        default=0.7,
        help="CSP denoise keep ratio of spatial components (0-1]",
    )
    parser.add_argument(
        "--denoise_csp_reg_eps",
        type=float,
        default=1e-6,
        help="Diagonal regularization for CSP covariance",
    )
    
    

    

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=6, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multiple gpus"
    )
    # parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

    # de-stationary projector params
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector (List)",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="number of hidden layers in projector",
    )

    args = parser.parse_args()
    if args.cif_snr_gain_csv:
        from models.cif_utils import load_cif_select_indices_from_csv

        args.cif_channel_mask = load_cif_select_indices_from_csv(
            args.cif_snr_gain_csv,
            splits=list(args.cif_snr_splits) if args.cif_snr_splits else None,
            min_delta=args.cif_snr_min_delta,
        )
        args.cif_select = True
        print(
            f"[CIF-select] Loaded {len(args.cif_channel_mask)} channels from "
            f"{args.cif_snr_gain_csv}: {args.cif_channel_mask}"
        )
    if args.cif_select and (not args.cif_channel_mask):
        raise ValueError(
            "cif_select=True requires --cif_channel_mask or --cif_snr_gain_csv"
        )
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        # print()

    print("Args in experiment:")
    print(args)
    
    # CUDA_VISIBLE_DEVICES=0,1,2,3


    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == "short_term_forecast":
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == "imputation":
        Exp = Exp_Imputation
    elif args.task_name == "anomaly_detection":
        Exp = Exp_Anomaly_Detection
    elif args.task_name == "classification":
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast



    if args.is_training:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            # if args.model != "TCN":
            #     torch.backends.cudnn.benchmark = False
            #     torch.backends.cudnn.deterministic = True


            # setting record of experiments
            args.seed = seed
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}".format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.seed,
            )

            log_dir = './log/' + args.task_name + "/" + args.model_id + "/" + args.model + '/' + setting + "/"
            
            # 创建日志目录（如果不存在）
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # 最终的日志文件路径
            log_file = os.path.join(log_dir, 'log.txt')

            # 现在你可以将日志文件传递给 logger 或其他操作

            # 设置 logger，并指定 mode='w' 以覆盖之前的日志文件
            logger = setup_logger(log_file, mode='w')

            exp = Exp(args)  # set experiments
            # print(
            #     ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            # )
            logger.info("Args in experiment:")
            logger.info(args)
            logger.info(f">>>>>>>training : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            # start training : classification_APAVA-Indep_Medformer_APAVA_ftM_sl96_ll48_pl96_dm128_nh8_el6_dl1_df256_fc1_ebtimeF_dtTrue_'Exp'_seed41>
            exp.train(setting,logger)

            # print(
            #     ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            # )
            logger.info(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting,logger=logger)
            # 关闭日志系统
            logging.shutdown()
            torch.cuda.empty_cache()
    else:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            args.seed = seed
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}".format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.seed,
            )



            exp = Exp(args)  # set experiments
            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            # logger.info(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

            exp.test(setting, test=1)
            torch.cuda.empty_cache()
