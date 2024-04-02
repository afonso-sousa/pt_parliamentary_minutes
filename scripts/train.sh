model_name="PORTULAN/albertina-100m-portuguese-ptpt-encoder" # "PORTULAN/albertina-1b5-portuguese-ptpt-encoder" # "PORTULAN/albertina-900m-portuguese-ptpt-encoder" # "neuralmind/bert-base-portuguese-cased" # "distilbert-base-multilingual-cased" "nb" "lr" "neuralmind/bert-base-portuguese-cased" "PORTULAN/albertina-1b5-portuguese-ptpt-encoder-256"
mode="train" # "train" "test" "attention_vis"
epochs=2
batch_size=32

CUDA_VISIBLE_DEVICES=0 python nlp_vote_prediction.py \
    --model $model_name \
    --mode $mode \
    --num_train_epochs $epochs \
    --train_batch_size $batch_size \
    --gradient_accumulation_steps 4 \
    # --with_lora \

