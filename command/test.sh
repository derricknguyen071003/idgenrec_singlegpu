datasets=${datasets}
item_gpu=${item_gpu:-2}
run_id=${run_id}
phase=${phase:-2}
run_type=${run_type:-2id2rec}
export CUDA_LAUNCH_BLOCKING=1
python ../src/main_generative.py --datasets $datasets \
  --use_diffusion 1 \
  --item_gpu $item_gpu \
  --phase $phase \
  --run_type $run_type \
  --rec_model_path /home/derrick/idgenrec_singlegpu/model/$datasets/$run_id/model_rec_round2_final.pt \
  --social_model_path /home/derrick/idgenrec_singlegpu/model/$datasets/$run_id/model_rec_round2_final.pt \
  --test_prompt seen:0 \
  --tasks sequential \
  --item_indexing generative \
  --run_id $run_id \
  --rounds 3 \
  --id_batch_size 12 \
  --rec_batch_size 64 \
  --prompt_file ../template/prompt.txt \
  --social_prompt_file ../template/prompt_social.txt \
  --sample_prompt 1 \
  --eval_batch_size 48 \
  --max_his 20  \
  --sample_num 1 \
  --train 0 \
  --alt_style id_first \
  --max_friend_his 20 \
