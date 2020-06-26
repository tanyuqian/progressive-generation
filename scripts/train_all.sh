python train.py --dataset cnn --prog_steps null-0.25-full --first_model gpt2-large
python train.py --dataset cnn --prog_steps null-0.2-0.25-full --first_model gpt2-large
python train.py --dataset cnn --prog_steps null-full --first_model gpt2
python train.py --dataset cnn --prog_steps null-full --first_model bart

python train.py --dataset wp --prog_steps null-0.25-full --first_model gpt2-large
python train.py --dataset wp --prog_steps null-0.2-0.25-full --first_model gpt2-large
python train.py --dataset wp --prog_steps null-full --first_model gpt2
python train.py --dataset wp --prog_steps null-full --first_model bart

# gpt2-large baselines need 4 GPUs to run, comment below if you don't have
python train.py --dataset cnn --prog_steps null-full --first_model gpt2-large
python train.py --dataset wp --prog_steps null-full --first_model gpt2-large