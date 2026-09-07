# Trail_snr

这个文件夹用于 **分析 CIF 方法的 Trial-averaged SNR**（以 `snr_iter_stats_seed*.txt` 为输入）。

## 输入数据（来自训练日志）

默认从 `../checkpoints/` 递归扫描 `snr_iter_stats_seed*.txt`（TSV 格式）。

其中常用列：
- `iter`
- `trial_delta_ch_dB_list`（每次迭代、每个“被改变的通道”的 SNR 增益列表，单位 dB）
- 或已计算好的 `trial_delta_ch_mean_dB`

## 输出

脚本会输出：
- `Trail_snr/trial_snr_by_seed.csv`：每个 run/seed 的 trial-averaged SNR
- `Trail_snr/trial_snr_summary_by_dataset.csv`：按数据集聚合（均值/方差/seed 数）

## 使用方法

在仓库根目录运行：

```bash
python Trail_snr/analyze_trial_snr.py \
  --root_dir ./checkpoints \
  --datasets APAVA ADFTD PTB TDBRAIN \
  --output_dir ./Trail_snr
```

如果你把 CIF 的实验都放在某个子目录下（例如 `./checkpoints/classification/CIF/`），可以用更小的 `--root_dir` 来加快扫描。

