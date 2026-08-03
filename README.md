# 新的聚类划分数据 已有train/val/test 

## 一、直接利用原始预训练模型推理
### 1、数据格式转化 CSV → .ms 文件
```bash
# 转测试集
python scripts/convert_csv_to_ms.py \
    --csv data/cluster/test.csv \
    --output-dir data/cluster_ms/test \
    --labels data/cluster/test_labels.tsv \
    --dataset cluster

# 转训练集
python scripts/convert_csv_to_ms.py \
    --csv data/cluster/train.csv \
    --output-dir data/cluster_ms/train \
    --labels data/cluster/train_labels.tsv \
    --dataset cluster

# 转验证集
python scripts/convert_csv_to_ms.py \
    --csv data/cluster/val.csv \
    --output-dir data/cluster_ms/val \
    --labels data/cluster/val_labels.tsv \
    --dataset cluster
```

### 2、子公式分配
```bash
# 测试集
python src/mist/subformulae/assign_subformulae.py \
  --spec-files data/cluster_ms/test/ \
  --labels-file data/cluster/test_labels.tsv \
  --output-dir data/cluster/subforms_test/ \
  --max-formulae 50 --num-workers 1

# 训练集
python src/mist/subformulae/assign_subformulae.py \
  --spec-files data/cluster_ms/train/ \
  --labels-file data/cluster/train_labels.tsv \
  --output-dir data/cluster/subforms_train/ \
  --max-formulae 50 --num-workers 1

# 验证集
python src/mist/subformulae/assign_subformulae.py \
  --spec-files data/cluster_ms/val/ \
  --labels-file data/cluster/val_labels.tsv \
  --output-dir data/cluster/subforms_val/ \
  --max-formulae 50 --num-workers 1
```

### 3、指纹预测
```bash
python src/mist/pred_fp.py \
  --num-workers 0 \
  --labels-file data/cluster/test_labels.tsv \
  --subform-folder data/cluster/subforms_test \
  --spec-folder data/cluster_ms/test/ \
  --dataset-name cluster_test \
  --model pretrained_models/mist_fp_canopus_pretrain.ckpt \
  --save-dir results/cluster/test
```

### 4、转表格 + 评估指标
```bash
python scripts/convert_pred_to_table.py \
  --pred results/cluster/test/fp_preds_cluster_test.p \
  --output results/cluster/test/fp_preds.tsv

python scripts/evaluate_fp.py \
  --pred results/cluster/test/fp_preds.tsv \
  --true data/cluster/test_molecular_fingerprints.tsv \
  --output results/cluster/test/hit_rate_0.5.tsv \
  --threshold 0.5

python scripts/evaluate_fp.py \
  --pred results/cluster/test/fp_preds.tsv \
  --true data/cluster/test_molecular_fingerprints.tsv \
  --output results/cluster/test/hit_rate_0.11.tsv \
  --threshold 0.11

python scripts/evaluate_fp.py \
  --pred results/cluster/test/fp_preds.tsv \
  --true data/cluster/test_molecular_fingerprints.tsv \
  --output results/cluster/test/hit_rate_0.2.tsv \
  --threshold 0.2
```


## 二、全量微调
### 1、训练集数据处理
```bash
# 生成切分文件：把三个 CSV 的 ID 合并成一个 split.tsv
python scripts/make_split_from_csv.py \
    --train data/cluster/train.csv \
    --val data/cluster/val.csv \
    --test data/cluster/test.csv \
    --output data/cluster/split.tsv

# 合并 前面的 train+val+test 标签
python scripts/merge_labels.py \
    --train data/cluster/train_labels.tsv \
    --val data/cluster/val_labels.tsv \
    --test data/cluster/test_labels.tsv \
    --output data/cluster/all_labels.tsv

```

### 2、全量微调训练 
```bash
# cosine 损失训练
python src/mist/train_mist.py \
  --spec-folder data/cluster_ms/all \
  --labels-file data/cluster/all_labels.tsv \
  --subform-folder data/cluster/subforms_all \
  --split-file data/cluster/split.tsv \
  --ckpt-file pretrained_models/mist_fp_canopus_pretrain.ckpt \
  --save-dir results/cluster/mist_ft \
  --fp-names morgan4096 \
  --form-embedder pos-cos \
  --hidden-size 256 \
  --loss-fn cosine \
  --refine-layers 4 \
  --peak-attn-layers 2 \
  --num-heads 8 \
  --pairwise-featurization \
  --no-diffs \
  --iterative-preds growing \
  --iterative-loss-weight 0.4 \
  --set-pooling cls \
  --spectra-dropout 0.1 \
  --learning-rate 0.0001 \
  --weight-decay 1e-07 \
  --batch-size 32 \
  --max-epochs 100 \
  --gpus 1 \
  --num-workers 4 \
  --magma-modulo 512

# 开数据增强 + 开学习率衰减
python src/mist/train_mist.py \
  --spec-folder data/cluster_ms/all \
  --labels-file data/cluster/all_labels.tsv \
  --subform-folder data/cluster/subforms_all \
  --split-file data/cluster/split.tsv \
  --ckpt-file pretrained_models/mist_fp_canopus_pretrain.ckpt \
  --save-dir results/cluster/mist_ft_data \
  --fp-names morgan4096 \
  --form-embedder pos-cos \
  --hidden-size 256 \
  --loss-fn cosine \
  --refine-layers 4 \
  --peak-attn-layers 2 \
  --num-heads 8 \
  --pairwise-featurization \
  --no-diffs \
  --iterative-preds growing \
  --iterative-loss-weight 0.4 \
  --set-pooling cls \
  --spectra-dropout 0.1 \
  --learning-rate 0.0001 \
  --weight-decay 1e-07 \
  --batch-size 32 \
  --max-epochs 100 \
  --gpus 1 \
  --num-workers 4 \
  --magma-modulo 512 
  --augment-data \              
  --augment-prob 1.0 \         
  --inten-prob 0.1 \           
  --remove-prob 0.5 \           
  --remove-weights exp \        
  --scheduler

# bce损失训练
python src/mist/train_mist.py \
  --spec-folder data/cluster_ms/all \
  --labels-file data/cluster/all_labels.tsv \
  --subform-folder data/cluster/subforms_all \
  --split-file data/cluster/split.tsv \
  --ckpt-file pretrained_models/mist_fp_canopus_pretrain.ckpt \
  --save-dir results/cluster/mist_ft_bce \
  --fp-names morgan4096 \
  --form-embedder pos-cos \
  --hidden-size 256 \
  --loss-fn bce \
  --refine-layers 4 \
  --peak-attn-layers 2 \
  --num-heads 8 \
  --pairwise-featurization \
  --no-diffs \
  --iterative-preds growing \
  --iterative-loss-weight 0.4 \
  --set-pooling cls \
  --spectra-dropout 0.1 \
  --learning-rate 0.0001 \
  --weight-decay 1e-07 \
  --batch-size 32 \
  --max-epochs 100 \
  --gpus 1 \
  --num-workers 4 \
  --magma-modulo 512

    
# 生成 cluster 训练集位频率权重
python scripts/compute_bit_weights.py \
  --labels data/cluster/train_labels.tsv \
  --output data/cluster/bit_weights_step.npy \
  --low-freq-weight 0.8 \
  --mid-freq-weight 1.5 \
  --high-freq-weight 3.0

# 有效化合物: 695
# 平均每位频率: 0.0115
# 最高频位: 0.9338
# 最低频位: 0.0000
# 权重范围: [0.80, 3.00]
#   频率 0.00~0.01: 2289 位, 平均权重 0.80
#   频率 0.01~0.05:  587 位, 平均权重 1.00
#   频率 0.05~0.10:   87 位, 平均权重 1.50
#   频率 0.10~0.20:   40 位, 平均权重 1.50
#   频率 0.20~0.50:   28 位, 平均权重 3.00
#   频率 0.50~1.00:   12 位, 平均权重 3.00

# 分段权重 bce 训练
python src/mist/train_mist_weighted.py \
  --spec-folder data/cluster_ms/all \
  --labels-file data/cluster/all_labels.tsv \
  --subform-folder data/cluster/subforms_all \
  --split-file data/cluster/split.tsv \
  --ckpt-file pretrained_models/mist_fp_canopus_pretrain.ckpt \
  --save-dir results/cluster/mist_ft_weighted \
  --fp-names morgan4096 \
  --form-embedder pos-cos \
  --hidden-size 256 \
  --loss-fn bce \
  --bit-weights data/cluster/bit_weights_step.npy \
  --refine-layers 4 \
  --peak-attn-layers 2 \
  --num-heads 8 \
  --pairwise-featurization \
  --no-diffs \
  --iterative-preds growing \
  --iterative-loss-weight 0.4 \
  --set-pooling cls \
  --spectra-dropout 0.1 \
  --learning-rate 0.0001 \
  --weight-decay 1e-07 \
  --batch-size 32 \
  --max-epochs 100 \
  --gpus 1 \
  --num-workers 4 \
  --magma-modulo 512
```

### 3、用新的权重推理
consine 损失训练
```bash
# 预测指纹
python src/mist/pred_fp.py \
  --num-workers 0 \
  --labels-file data/cluster/test_labels.tsv \
  --subform-folder data/cluster/subforms_test \
  --spec-folder data/cluster_ms/test \
  --dataset-name mist_ft \
  --model-ckpt results/cluster/mist_ft/split/best.ckpt \
  --save-dir results/cluster/mist_ft

# 格式转换：.p -> tsv
python scripts/convert_pred_to_table.py \
  --pred results/cluster/mist_ft/fp_preds_mist_ft.p \
  --output results/cluster/mist_ft/fp_preds.tsv

# 评估
python scripts/evaluate_fp.py \
    --pred results/cluster/mist_ft/fp_preds.tsv \
    --true data/cluster/test_molecular_fingerprints.tsv \
    --output results/cluster/mist_ft/hit_rate_0.5.tsv \
    --threshold 0.5
python scripts/evaluate_fp.py \
    --pred results/cluster/mist_ft/fp_preds.tsv \
    --true data/cluster/test_molecular_fingerprints.tsv \
    --output results/cluster/mist_ft/hit_rate_0.11.tsv \
    --threshold 0.11
python scripts/evaluate_fp.py \
    --pred results/cluster/mist_ft/fp_preds.tsv \
    --true data/cluster/test_molecular_fingerprints.tsv \
    --output results/cluster/mist_ft/hit_rate_0.2.tsv \
    --threshold 0.2
```

consine损失 + 数据增强 训练 
```bash
# 预测指纹
python src/mist/pred_fp.py \
  --num-workers 0 \
  --labels-file data/cluster/test_labels.tsv \
  --subform-folder data/cluster/subforms_test \
  --spec-folder data/cluster_ms/test \
  --dataset-name mist_ft \
  --model-ckpt results/cluster/mist_ft_data/split/best.ckpt \
  --save-dir results/cluster/mist_ft_data

# 格式转换：.p -> tsv
python scripts/convert_pred_to_table.py \
  --pred results/cluster/mist_ft_data/fp_preds_mist_ft.p \
  --output results/cluster/mist_ft_data/fp_preds.tsv

# 评估
python scripts/evaluate_fp.py \
    --pred results/cluster/mist_ft_data/fp_preds.tsv \
    --true data/cluster/test_molecular_fingerprints.tsv \
    --output results/cluster/mist_ft_data/hit_rate_0.5.tsv \
    --threshold 0.5
python scripts/evaluate_fp.py \
    --pred results/cluster/mist_ft_data/fp_preds.tsv \
    --true data/cluster/test_molecular_fingerprints.tsv \
    --output results/cluster/mist_ft_data/hit_rate_0.11.tsv \
    --threshold 0.11
python scripts/evaluate_fp.py \
    --pred results/cluster/mist_ft_data/fp_preds.tsv \
    --true data/cluster/test_molecular_fingerprints.tsv \
    --output results/cluster/mist_ft_data/hit_rate_0.2.tsv \
    --threshold 0.2
```



bce 损失训练
```bash
# 预测指纹
python src/mist/pred_fp.py \
  --num-workers 0 \
  --labels-file data/cluster/test_labels.tsv \
  --subform-folder data/cluster/subforms_test \
  --spec-folder data/cluster_ms/test \
  --dataset-name mist_ft \
  --model-ckpt results/cluster/mist_ft_bce/split/best.ckpt \
  --save-dir results/cluster/mist_ft_bce

# 格式转换：.p -> tsv
python scripts/convert_pred_to_table.py \
  --pred results/cluster/mist_ft_bce/fp_preds_mist_ft.p \
  --output results/cluster/mist_ft_bce/fp_preds.tsv

# 评估
python scripts/evaluate_fp.py \
    --pred results/cluster/mist_ft_bce/fp_preds.tsv \
    --true data/cluster/test_molecular_fingerprints.tsv \
    --output results/cluster/mist_ft_bce/hit_rate_0.5.tsv \
    --threshold 0.5
python scripts/evaluate_fp.py \
    --pred results/cluster/mist_ft_bce/fp_preds.tsv \
    --true data/cluster/test_molecular_fingerprints.tsv \
    --output results/cluster/mist_ft_bce/hit_rate_0.11.tsv \
    --threshold 0.11
python scripts/evaluate_fp.py \
    --pred results/cluster/mist_ft_bce/fp_preds.tsv \
    --true data/cluster/test_molecular_fingerprints.tsv \
    --output results/cluster/mist_ft_bce/hit_rate_0.2.tsv \
    --threshold 0.2
```

bce 权重损失训练
```bash
# 预测指纹
python src/mist/pred_fp.py \
  --num-workers 0 \
  --labels-file data/cluster/test_labels.tsv \
  --subform-folder data/cluster/subforms_test \
  --spec-folder data/cluster_ms/test \
  --dataset-name mist_ft \
  --model-ckpt results/cluster/mist_ft_weighted/split/best.ckpt \
  --save-dir results/cluster/mist_ft_weighted

# 格式转换：.p -> tsv
python scripts/convert_pred_to_table.py \
  --pred results/cluster/mist_ft_weighted/fp_preds_mist_ft.p \
  --output results/cluster/mist_ft_weighted/fp_preds.tsv

# 评估
python scripts/evaluate_fp.py \
    --pred results/cluster/mist_ft_weighted/fp_preds.tsv \
    --true data/cluster/test_molecular_fingerprints.tsv \
    --output results/cluster/mist_ft_weighted/hit_rate_0.5.tsv \
    --threshold 0.5
python scripts/evaluate_fp.py \
    --pred results/cluster/mist_ft_weighted/fp_preds.tsv \
    --true data/cluster/test_molecular_fingerprints.tsv \
    --output results/cluster/mist_ft_weighted/hit_rate_0.11.tsv \
    --threshold 0.11
python scripts/evaluate_fp.py \
    --pred results/cluster/mist_ft_weighted/fp_preds.tsv \
    --true data/cluster/test_molecular_fingerprints.tsv \
    --output results/cluster/mist_ft_weighted/hit_rate_0.2.tsv \
    --threshold 0.2
```
