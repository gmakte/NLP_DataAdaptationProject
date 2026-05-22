# NLP Data Adaptation Project

## Project Resources

- The FIN5, FIN3, and control datasets are stored in the [`data/`](data) folder.
- Predictions generated when the model was evaluated on FIN3 are saved in the [`predictions/`](predictions) folder.
- Evaluation outputs are saved in the [`results/`](results) folder.
- Synthetic contracts are saved in [`synthetic/contracts/`](synthetic/contracts).
- Jupyter notebooks for visualization, exploration, and error analysis are stored in the repository as well.
- Environment setup is documented in `requirements.txt` and `environemnt.yaml`.
- Prompts for contracts generation are available in [`synthetic/`](synthetic/).

## Runtime Notes

- Synthetic contract generation with the Mistral requires about 15 GB of memory, so it should be run on HPC rather than locally.
- Model training and prediction generation can be done locally, but HPC is recommended if available because these runs can take more than 6 hours.

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
python model_train.py data/FIN5_train.txt model1 400 --learning_rate 2e-5 --num_train_epochs 8 --batch_size 15
```

What the script does:

- loads and parses the training and development IOB2 files
- tokenizes the data and aligns NER labels to subword tokens
- fine-tunes a token-classification model
- saves the trained model and tokenizer to `model_dir`
- writes training losses to `results/`

## How to Run Predictions

The prediction entry point is [`predictions.py`](predictions.py). Run it from the project root with three positional arguments:

- `input_path`: path to the input IOB2-style file
- `output_path`: path where the predictions will be written
- `model_dir`: directory containing the trained model and tokenizer

Example:

```bash
python predictions.py data/FIN3_fixed.txt predictions/test_predictions.txt model1
```

The input file should contain tokenized sentences, with sentences separated by blank lines. The script loads the trained model, predicts labels for each token, and writes the output as `token_id`, `token`, and `label` columns separated by tabs.

## How to Evaluate Predictions

The evaluation entry point is [`main.py`](main.py). Use `--mode evaluate` to compare a prediction file against a reference file and save the metrics.

Required inputs:

- `--pred_path`: path to the prediction file
- `--ref_path`: path to the gold/reference file

Optional outputs:

- `--output_dir`: directory where evaluation results are saved, default `results/`
- `--output_file`: JSON file for the metrics, default `results_mixed_6.json`
- `--qualitative_file`: tab-separated sentence-level comparison file, default `results_mixed_6.txt`

Example:

```bash
python main.py --mode evaluate --pred_path predictions/test_predictions.txt --ref_path data/FIN3_fixed.txt --output_dir results --output_file results_test.json --qualitative_file results_test.txt
```

This command writes a JSON report with seqeval, span-level, and entity-level metrics, and it also saves a qualitative file with aligned gold and predicted labels for each token.

## How to Generate Synthetic Contracts

The synthetic data entry point is [`synthetic/generate_synth_data_v2.py`](synthetic/generate_synth_data_v2.py). Run it from the `synthetic/` directory, because it loads its prompt templates with relative paths.

Required arguments:

- `--output_file`: name of the generated contract file

Optional arguments:

- `--model_name`: model preset to use, default `mistral`
- `--quantized`: load the model in 4-bit mode for lower memory use
- `--sectioned`: generate the contract section by section instead of in one pass
- `--max_new_tokens`: maximum number of tokens to generate, default `2000`
- `--output_dir`: output directory, default `contracts/`
- `--plan_temperature`: temperature for plan generation, default `0.9`
- `--section_temperature`: temperature for contract generation, default `0.6`

Example:

```bash
cd synthetic
python generate_synth_data_v2.py --output_file generated_contract.txt --model_name mistral --sectioned
```

This command generates a synthetic contract and saves it to `synthetic/contracts/generated_contract.txt` by default. Supported model presets are `mistral`, `llama`, `qwen`, `deepseek` and `gemma`.

## How to Annotate Synthetic Contracts

The annotation entry point is [`annotation/annotate_data.py`](annotation/annotate_data.py). Run it from the project root so its relative paths resolve correctly.

For synthetic contracts, use `--mode annotate` and point `--input_data` to the generated contract file.

Required arguments:

- `--mode annotate`: annotate a contract file
- `--output_file`: name of the annotated output file

Useful optional arguments:

- `--input_data`: path to the input contract file, default `synthetic/contracts_train`
- `--output_dir`: directory for annotated files, default `annotation/contracts`
- `--model_name`: model preset to use, default `mistral`
- `--quantized`: load the model in 4-bit mode for lower memory use
- `--max_chunk_size`: maximum number of words per chunk, default `50`
- `--temperature`: generation temperature, default `0`
- `--load_chunks`: reuse a previously saved `annotation/chunks.json`

Example:

```bash
python annotation/annotate_data.py --mode annotate --input_data synthetic/contracts/generated_contract.txt --output_dir annotation/annotated_contracts --output_file generated_contract_annotated.txt --model_name mistral
```

This command splits the contract into chunks, sends each chunk to the model for token-level annotation, and writes the final annotated file to the chosen output directory. The script also saves the chunk structure to `annotation/chunks.json` for reuse unless you run in `repair` mode.
