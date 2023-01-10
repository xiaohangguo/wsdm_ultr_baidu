python -u submit.py --emb_dim 768 --nlayer 3 --nhead 12 \
 --ntokens 22872 \
 --init_parameters 'save_model/save_steps5000_1.27456_0.592206.model' \
 --test_annotate_path "test_2/wsdm_test_2_all.txt" \
 --eval_batch_size 50