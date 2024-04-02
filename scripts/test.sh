# model_name="neuralmind/bert-base-portuguese-cased" # "distilbert-base-multilingual-cased" "nb" "lr" "neuralmind/bert-base-portuguese-cased" "PORTULAN/albertina-1b5-portuguese-ptpt-encoder-256"
mode="test" # "train" "test" "attention_vis"
# model_name="model_results/neuralmind/bert-base-portuguese-cased/checkpoint-584"
# model_name="model_results/PORTULAN/albertina-900m-portuguese-ptpt-encoder/checkpoint-1168"
# model_name="model_results/PORTULAN/albertina-1b5-portuguese-ptpt-encoder/checkpoint-582"
model_name="model_results/PORTULAN/albertina-100m-portuguese-ptpt-encoder/checkpoint-72"

CUDA_VISIBLE_DEVICES=1 python nlp_vote_prediction.py \
    --model $model_name \
    --mode $mode
    # --with_lora

