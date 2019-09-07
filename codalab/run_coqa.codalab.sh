start_time=`date +%s`

alias python=python3
pip install -U sentencepiece

git clone --recurse-submodules https://github.com/stevezheng23/mrc_tf.git

cd mrc_tf

mkdir model
mkdir model/xlnet
wget -P model/xlnet https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
unzip model/xlnet/cased_L-24_H-1024_A-16.zip -d model/xlnet/
mv model/xlnet/xlnet_cased_L-24_H-1024_A-16 model/xlnet/cased_L-24_H-1024_A-16
rm model/xlnet/cased_L-24_H-1024_A-16.zip

mkdir data
mkdir data/coqa
cp ../coqa-dev-v1.0.json data/coqa/dev-v1.0.json

mkdir output
mkdir output/coqa
mkdir output/coqa/data
wget -P output/coqa https://storage.googleapis.com/coqa/coqa_cased_L-12_H-768_A-12.zip
unzip output/coqa/coqa_cased_L-12_H-768_A-12.zip -d output/coqa/
mv output/coqa/coqa_cased_L-12_H-768_A-12 output/coqa/checkpoint
rm output/coqa/coqa_cased_L-12_H-768_A-12.zip

CUDA_VISIBLE_DEVICES=0 python run_coqa.py \
--spiece_model_file=model/xlnet/cased_L-24_H-1024_A-16/spiece.model \
--model_config_path=model/xlnet/cased_L-24_H-1024_A-16/xlnet_config.json \
--init_checkpoint=model/xlnet/cased_L-24_H-1024_A-16/xlnet_model.ckpt \
--task_name='v1.0' \
--random_seed=1000 \
--predict_tag='v1.0' \
--lower_case=false \
--data_dir=data/coqa/ \
--output_dir=output/coqa/data \
--model_dir=output/coqa/checkpoint \
--export_dir=output/coqa/export \
--max_seq_length=512 \
--max_query_length=64 \
--max_answer_length=16 \
--train_batch_size=48 \
--predict_batch_size=16 \
--num_hosts=1 \
--num_core_per_host=1 \
--learning_rate=3e-5 \
--train_steps=15000 \
--warmup_steps=0 \
--save_steps=3000 \
--do_train=false \
--do_predict=true \
--do_export=false \
--overwrite_data=false

python tool/convert_coqa.py \
--input_file=output/coqa/data/predict.v1.0.summary.json \
--output_file=output/coqa/data/predict.v1.0.span.json \
--answer_threshold=0.35

python tool/eval_coqa.py \
--data-file=data/coqa/dev-v1.0.json \
--pred-file=output/coqa/data/predict.v1.0.span.json \
>> output/coqa/data/predict.v1.0.eval.json

cp output/coqa/data/predict.v1.0.eval.json ../coqa-dev-v1.0.eval.json

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.
