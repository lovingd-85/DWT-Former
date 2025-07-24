export CUDA_VISIBLE_DEVICES=1
##Competation
# python -u ../run.py \
#   --root_path ../dataset/ \
#   --data_path Competation_Data.csv \
#   --data custom \
#   --freq h \
#   --features MS \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 24 \
#   --c_in 9 \
#   --c_out 9 \
#   --input 9 \
#   --batch_size 512 \
#   --sampling_layers 2 \
#   --patience 3 \
#   --use_gpu True \

## mydata
python -u ../run.py \
  --root_path ../dataset/ \
  --data_path mydata.csv \
  --data custom \
  --freq t \
  --features MS \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 24 \
  --c_in 17 \
  --c_out 17 \
  --input 17 \
  --batch_size 512 \
  --sampling_layers 2 \
  --patience 3 \
  --use_gpu True \