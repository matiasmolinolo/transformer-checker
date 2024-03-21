# Transformer Checker

This project is a tool to visualize and check neural activations and attention patterns in transformer models that classify context-free languages.

## Generating Data
We provide a tool to generate data for training the transformer model. The tool is located in the `dyck-k-generation` directory. To use the tool, run the following command:

```bash
python -m dyck_k_generator.generator --n n --k k --max_length max_length --balanced balanced --file
```

where `n` is the number of samples, `k` is the Dyck-k language, `max_length` is the maximum length of the strings, `balanced` is a float between 0 and 1 that represents the percentage of balanced strings, and `file` is flag that represents whether to save the data to a file or return it to a variable.

Default values are `n=500_000`, `k=3`, `max_length=1025` and `balanced=0.5`.
