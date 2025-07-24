export CUDA_VISIBLE_DEVICES=1
python -u ../run.py \
  --root_path ../dataset/ \
  --data_path  CC-PV.csv \
  --data custom \
  --freq t \
  --features MS \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 24 \
  --c_in 8 \
  --c_out 8 \
  --input 8 \
  --batch_size 62 \
  --sampling_layers 2 \
  --patience 10 \
  --use_gpu True \