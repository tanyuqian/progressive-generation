python evaluate.py --dataset cnn --prog_steps null-0.25-full --first_model gpt2-large
python evaluate.py --dataset cnn --prog_steps null-0.2-0.25-full --first_model gpt2-large
python evaluate.py --dataset cnn --prog_steps null-full --first_model gpt2
python evaluate.py --dataset cnn --prog_steps null-full --first_model bart

python evaluate.py --dataset wp --prog_steps null-0.25-full --first_model bart
python evaluate.py --dataset wp --prog_steps null-0.2-0.25-full --first_model bart
python evaluate.py --dataset wp --prog_steps null-full --first_model gpt2
python evaluate.py --dataset wp --prog_steps null-full --first_model bart

python evaluate.py --dataset cnn --prog_steps null-full --first_model gpt2-large
python evaluate.py --dataset wp --prog_steps null-full --first_model gpt2-large