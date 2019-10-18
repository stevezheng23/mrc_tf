while [[ $# -gt 0 ]]
  do
    key="$1"
    case $key in
      -i|--inputfile)
      INPUTFILE="$2"
      shift
      shift
      ;;
      -o|--outputfile)
      OUTPUTFILE="$2"
      shift
      shift
      ;;
    esac
  done

echo "input file        = ${INPUTFILE}"
echo "output file       = ${OUTPUTFILE}"

start_time=`date +%s`

alias python=python3

git clone --recurse-submodules https://github.com/stevezheng23/mrc_tf.git

cd mrc_tf

mkdir model
mkdir model/xlnet
wget -P model/xlnet https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
unzip model/xlnet/cased_L-24_H-1024_A-16.zip -d model/xlnet/
mv model/xlnet/xlnet_cased_L-24_H-1024_A-16 model/xlnet/cased_L-24_H-1024_A-16
rm model/xlnet/cased_L-24_H-1024_A-16.zip

mkdir data
mkdir data/quac
cp ../${INPUTFILE} data/quac/dev-v0.2.json

mkdir output
mkdir output/quac
mkdir output/quac/data
wget -P output/quac https://storage.googleapis.com/mrc_data/quac/quac_cased_L-24_H-1024_A-16.zip
unzip output/quac/quac_cased_L-24_H-1024_A-16.zip -d output/quac/
mv output/quac/quac_cased_L-24_H-1024_A-16 output/quac/checkpoint
rm output/quac/quac_cased_L-24_H-1024_A-16.zip

CUDA_VISIBLE_DEVICES=0 python run_quac.py \
--spiece_model_file=model/xlnet/cased_L-24_H-1024_A-16/spiece.model \
--model_config_path=model/xlnet/cased_L-24_H-1024_A-16/xlnet_config.json \
--init_checkpoint=model/xlnet/cased_L-24_H-1024_A-16/xlnet_model.ckpt \
--task_name='v0.2' \
--random_seed=1000 \
--predict_tag='v0.2' \
--lower_case=false \
--data_dir=data/quac/ \
--output_dir=output/quac/data \
--model_dir=output/quac/checkpoint \
--export_dir=output/quac/export \
--num_turn=-1 \
--max_seq_length=512 \
--max_query_length=192 \
--max_answer_length=32 \
--train_batch_size=48 \
--predict_batch_size=12 \
--num_hosts=1 \
--num_core_per_host=1 \
--learning_rate=2e-5 \
--train_steps=8000 \
--warmup_steps=1000 \
--save_steps=1000 \
--do_train=false \
--do_predict=true \
--do_export=false \
--overwrite_data=false

python tool/convert_quac.py \
--input_file=output/quac/data/predict.v0.2.summary.json \
--output_file=output/quac/data/predict.v0.2.span.json

python tool/eval_quac.py \
--val_file=data/quac/dev-v0.2.json \
--model_output=output/quac/data/predict.v0.2.span.json \
--o=output/quac/data/predict.v0.2.eval.json

rm -r model/xlnet/
rm -r output/quac/checkpoint/

cp output/quac/data/predict.v0.2.span.json ../${OUTPUTFILE}

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.
