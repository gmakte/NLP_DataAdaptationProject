# NLP Data Adaptation Project

## How to Train the Model

The training entry point is [`model_train.py`](model_train.py). Run it from the project root and provide three required arguments:

- `path_train`: path to the training IOB2 file
- `model_dir`: directory where the trained model and tokenizer will be saved
- `max_length`: maximum tokenization length

Optional arguments:

- `--path_dev`: development IOB2 file, default `data/FIN5_dev.txt`
- `--path_test`: test IOB2 file, default `data/FIN3_fixed.txt`
- `--model_name`: Hugging Face model name, default `google-bert/bert-base-cased`
- `--learning_rate`: learning rate, default `2e-5`
- `--num_train_epochs`: number of epochs, default `8`
- `--batch_size`: batch size, default `15`

Example:

```bash
python model_train.py data/FIN5_train.txt model1 256 --path_dev data/FIN5_dev.txt --path_test data/FIN3_fixed.txt --model_name google-bert/bert-base-cased --learning_rate 2e-5 --num_train_epochs 8 --batch_size 15
```

What the script does:

- loads and parses the training and development IOB2 files
- tokenizes the data and aligns NER labels to subword tokens
- fine-tunes a token-classification model
- saves the trained model and tokenizer to `model_dir`
- writes training losses to `results/training_losses_FIN5full_256.txt`

Note: the script currently has `only_trainning = True`, so it trains the model and skips test-set prediction unless that flag is changed to `False` in [`model_train.py`](model_train.py).
