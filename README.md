# Progressive Generation

Code to be cleaned up soon...

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

## Train
```bash
python train.py \
    --dataset [cnn/wp] \
    --prog_steps null-{...}-full \
    --first_model [bart/gpt2/gpt2-large]
```
* ```--first_model``` specifies the type of the first-stage model.

The training log will be stored in ```training_logs/{your setting}/```:
* ```training_logs/{setting}/log.txt```: evaluation loss of each checkpoint.
* ```training_logs/{setting}/ckpt_gens/step{}.txt```: ~10 generation examples on dev set of each checkpoint.
* ```training_logs/{setting}/best_model.pt```: best checkpoint model according to evaluation loss.

Check [scripts/train_all.sh](scripts/train_all.sh) for all commands for training.

## Generate
```bash
python generate.py \
    --dataset [cnn/wp] \
    --prog_steps null-{...}-full \
    --first_model [bart/gpt2/gpt2-large]
```
Generated texts will be stored in ```generated_texts/{your setting}/```:
* ```generated_texts/{setting}/gen.txt```: generation log.
* ```generated_texts/{setting}/gen.pickle```: all generated texts stored into a pickle file.

Check [scripts/gen_all.sh](scripts/gen_all.sh) for all commands for generation.

## Evaluate
```bash
python evaluate.py \
    --dataset [cnn/wp] \
    --prog_steps null-{...}-full \
    --first_model [bart/gpt2/gpt2-large]
```

Check [scripts/eval_all.sh](scripts/eval_all.sh) for all commands for evaluation.

## Present Results
```bash
python present_eval_results.py \
    --dataset [cnn/wp] \
    --metric [ms_jaccard/frechet_bert_distance/tfidf_distance/forward_backward_bleu]
```

Check [scripts/present_all.sh](scripts/present_all.sh) for all commands for presenting.

## Results
For the simplicity to run our code, we reduce the test set to 
1K examples (numbers in our paper are from 5K examples). 
If you want to change it back to 5K, see [download/download_cnn.py](download/download_cnn.py#L128) 
and [download/download_writing_prompts.py](download/download_writing_prompts.py#L23).

Results of 1K examples are as below. 
The superiority of our progressive models is already significant.

### Dataset: CNN
| Metric   | BART  | GPT2-Small | GPT2-Large | ProGeT-2 | ProGeT-3 |
|----------|-------|------------|------------|----------|----------|
| HA-BLEU2 | 75.42 | 75.34      | 75.44      | 76.13    | **76.54**    |
| HA-BLEU3 | 51.87 | 51.21      | 51.50      | 52.66    | **53.20**    |
| HA-BLEU4 | 31.96 | 31.16      | 31.36      | 32.47    | **33.01**    |
| HA-BLEU5 | 18.68 | 18.05      | 18.14      | 19.00    | **19.42**    |
| MSJ2     | 49.45 | 50.66      | 50.48      | 51.93    | **52.46**    |
| MSJ3     | 29.59 | 29.87      | 30.03      | 31.37    | **31.84**    |
| MSJ4     | 16.67 | 16.53      | 16.71      | 17.72    | **18.09**    |
| MSJ5     | 9.24  | 9.02       | 9.16       | 9.83     | **10.08**    |
| FBD-S    | 6.70  | 6.42       | 6.27       | **6.15**     | 6.12     |
| FBD-M    | 21.85 | 18.60      | 15.93      | 15.96    | **15.53**    |
| FBD-D    | 48.50 | 41.82      | 34.68      | 33.04    | **32.57**    |
| TID      | 9.72  | 9.68       | 7.82       | 6.74     | **5.83**     |
### Dataset: WritingPrompts
| Metric   | BART  | GPT2-Small | GPT2-Large | ProGeT-2 | ProGeT-3 |
|----------|-------|------------|------------|----------|----------|
| HA-BLEU2 | 79.95 | 78.55      | 78.49      | **80.70**    | 80.09    |
| HA-BLEU3 | 58.38 | 55.95      | 55.95      | **59.05**    | 58.51    |
| HA-BLEU4 | 36.84 | 34.06      | 34.21      | **37.24**    | 36.82    |
| HA-BLEU5 | 20.71 | 18.34      | 18.45      | **20.77**    | 20.48    |
| MSJ2     | 57.51 | 57.58      | 54.85      | **58.28**    | 57.43    |
| MSJ3     | 37.10 | 36.00      | 34.41      | **37.64**    | 37.26    |
| MSJ4     | 21.78 | 20.37      | 19.60      | **22.09**    | 21.88    |
| MSJ5     | 12.14 | 10.89      | 10.55      | **12.20**    | 12.08    |
| FBD-S    | 3.72  | 3.03       | 3.67       | **2.67**     | 3.17     |
| FBD-M    | 19.83 | 18.99      | 18.40      | **17.41**    | 18.64    |
| FBD-D    | 44.04 | 42.53      | 41.07      | **37.95**    | 40.71    |
| TID      | 4.74  | 5.00       | 5.84       | **3.99**     | 4.08     |