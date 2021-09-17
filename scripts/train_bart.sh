code_path=~/BART_github/code
output_path=~/BART_github/Output/
models_path=~/BART_github/Models/
#####################################################################################
course_dir=cmirror

################# Train ########################
python ${code_path}/run_summarization.py --model_name_or_path sshleifer/distilbart-cnn-6-6 --do_train --train_file ${code_path}/data/${course_dir}/train.csv --validation_file ${code_path}/data/${course_dir}/val.csv --test_file ${code_path}/data/${course_dir}/test.csv --output_dir ${models_path}/${course_dir}_a --overwrite_output_dir --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --predict_with_generate --num_train_epochs 15 --save_strategy epoch --seed 777 --num_gpus 4


################# Generate summaries for validation data using all checkpoints ########################
for i in {0..14}
do
python ${code_path}/run_summarization.py --model_name_or_path ${models_path}/${course_dir}_a/checkpoint-${i} --do_predict --train_file ${code_path}/data/${course_dir}/train.csv --validation_file ${code_path}/data/${course_dir}/val.csv --test_file ${code_path}/data/${course_dir}/val.csv --output_dir ${output_path}/${course_dir}_a/Prediction_${i} --overwrite_output_dir --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --predict_with_generate --num_train_epochs 15 --save_strategy epoch --seed 777 --num_gpus 4
done


###### Do final testing using best checkpoint ####################################
python ${code_path}/run_summarization.py --model_name_or_path ${models_path}/${course_dir}_a/checkpoint-best --do_predict --train_file ${code_path}/data/${course_dir}/train.csv --validation_file ${code_path}/data/${course_dir}/val.csv --test_file ${code_path}/data/${course_dir}/test.csv --output_dir ${output_path}/${course_dir}_a/Prediction_best --overwrite_output_dir --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --predict_with_generate --num_train_epochs 15 --save_strategy epoch --seed 777 --num_gpus 4

