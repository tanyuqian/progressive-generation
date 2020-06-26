# Progressive Generation

## Requirements
```
torch==1.2.0
transformers==2.5.1
fairseq==0.9.0
```
It needs at least 4 GPUs on your device if you want to finetune GPT2-Large baseline, otherwise 2 GPUs are enough.  

## Download Data
```bash
python download/download_cnn.py
python download/download_writing_prompts.py
```

## Code
### Train
```bash
python train.py \
    --dataset [cnn/wp] \
    --prog_steps null-0.2-0.25-full \
    --first_model [bart/gpt2/gpt2-large]
```
The training log will be stored in ```training_logs/{your setting}/```:
* ```training_logs/{setting}/log.txt```: evaluation loss of each checkpoint.
* ```training_logs/{setting}/ckpt_gens/step{}.txt```: ~10 generation examples on dev set of each checkpoint.
* ```training_logs/{setting}/best_model.pt```: best checkpoint model according to evaluation loss.

### Generate
```bash
python generate.py \
    --dataset [cnn/wp] \
    --prog_steps null-0.2-0.25-full \
    --first_model [bart/gpt2/gpt2-large]
```
Generated texts will be stored in ```generated_texts/{your setting}/```:
* ```generated_texts/{setting}/gen.txt```: generation log.
* ```generated_texts/{setting}/gen.pickle```: all generated texts stored into a pickle file.

### Evaluate
```bash
python generate.py \
    --dataset [cnn/wp] \
    --prog_steps null-0.2-0.25-full \
    --first_model [bart/gpt2/gpt2-large]
```

### Show Results
```bash
python present_eval_results.py \
    --dataset [cnn/wp] \
    --metric [ms_jaccard/frechet_bert_distance/tfidf_distance/forward_backward_bleu]
```

## Results
