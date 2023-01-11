python -u pretrain.py --init_parameters baidu_ultr_3l_12h_768e.model \
--emb_dim 768 --nlayer 3 --nhead 12 --dropout 0.1 --buffer_size 40 --eval_batch_size 20 \
--valid_click_path data/click_data/test_click.data.gz --save_step 5000 --eval_step 500 \
--n_queries_for_each_gpu 4 --num_candidates 20 \
--log_interval 100 \
--valid_annotate_path ./data/annotate_data/valid_data.txt 
