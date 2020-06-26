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
python evaluate.py \
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
python generate.py \
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
1K examples (numbers in our paper is from 5K examples). 
If you want to change it back to 5K, see ```download/download_cnn.py::Line128``` 
and ```download/download_writing_prompts.py::Line23```.

Results of 1K examples are as below. 
The superiority of our progressive models is already significant.

### Dataset: CNN
| Metric   | BART  | GPT2-Small | GPT2-Large | ProGeT-2 | ProGeT-3 |
|----------|-------|------------|------------|----------|----------|
| HA-BLEU2 |       |            |            |          |          |
| HA-BLEU3 |       |            |            |          |          |
| HA-BLEU4 |       |            |            |          |          |
| HA-BLEU5 |       |            |            |          |          |
| MSJ2     | 49.45 | 50.66      | 50.48      | 51.93    | **52.46**    |
| MSJ3     | 29.59 | 29.87      | 30.03      | 31.37    | **31.84**    |
| MSJ4     | 16.67 | 16.53      | 16.71      | 17.72    | **18.09**    |
| MSJ5     | 9.24  | 9.02       | 9.16       | 9.83     | **10.08**    |
| FBD-S    | 6.70  | 6.42       | 6.27       | 6.15     | **6.12**     |
| FBD-M    | 21.85 | 18.60      | 15.93      | 15.96    | **15.53**    |
| FBD-D    | 48.50 | 41.82      | 34.68      | 33.04    | **32.57**    |
| TID      | 9.72  | 9.68       | 7.82       | 6.74     | **5.83**     |

### Dataset: WritingPrompts
| Metric   | BART | GPT2-Small | GPT2-Large | ProGeT-2 | ProGeT-3 |
|----------|------|------------|------------|----------|----------|
| HA-BLEU2 |      |            |            |          |          |
| HA-BLEU3 |      |            |            |          |          |
| HA-BLEU4 |      |            |            |          |          |
| HA-BLEU5 |      |            |            |          |          |
| MSJ2     |      |            |            |          |          |
| MSJ3     |      |            |            |          |          |
| MSJ4     |      |            |            |          |          |
| MSJ5     |      |            |            |          |          |
| FBD-S    |      |            |            |          |          |
| FBD-M    |      |            |            |          |          |
| FBD-D    |      |            |            |          |          |
| TID      |      |            |            |          |          |