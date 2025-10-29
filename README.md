# Efficient Transformer Fine-Tuning with LoRA

This project, developed for an M.S. Machine Learning course, implements Low-Rank Adaptation (LoRA) from scratch to efficiently fine-tune Transformer models. It explores the trade-offs between performance and parameter efficiency by comparing three methods: training a small BERT from scratch, fully fine-tuning a pre-trained TinyBERT, and applying our custom LoRA implementation to TinyBERT for a text classification task on the Consumer Complaint dataset.

## Features

* **Custom LoRA Implementation:** A from-scratch implementation of the LoRA technique in PyTorch (`src/lora.py`), demonstrating a deep understanding of PEFT.
* **Modular Experiment Framework:** The project is structured with separate, parameterized scripts for data preprocessing (`preprocess.py`) and training each model (`train_bert_scratch.py`, `train_full_finetune.py`, `train_lora.py`).
* **Comparative Analysis:** Provides a direct comparison of three distinct training strategies, analyzing both final F1-score and the number of trainable parameters.
* **Reproducibility:** Includes a `requirements.txt` file and a `run_experiments.ipynb` notebook to easily replicate the entire experimental pipeline and visualize the results.

## Core Concepts & Techniques

* **Parameter-Efficient Fine-Tuning (PEFT):** The overarching strategy of fine-tuning large models by updating only a small subset of their parameters.
* **Low-Rank Adaptation (LoRA):** The specific PEFT technique implemented, which injects trainable low-rank matrices into the model.
* **Transformers (BERT):** The underlying model architecture used for the text classification task.
* **Text Classification (NLP):** The downstream task of categorizing consumer complaints into one of nine product categories.
* **Knowledge Distillation:** The concept behind TinyBERT, the pre-trained model we use as our base for fine-tuning.

---

## How It Works

This project investigates the challenge of fine-tuning large pre-trained language models, which is often computationally prohibitive. Our goal is to demonstrate that Parameter-Efficient Fine-Tuning (PEFT) methods, specifically LoRA, can achieve performance nearly identical to full fine-tuning while training only a tiny fraction of the parameters.

### 1. Core Implementation: Low-Rank Adaptation (LoRA)

The central hypothesis of LoRA is that the *change* in weights during model adaptation ($\Delta W$) has a low "intrinsic rank." This means the update can be effectively represented by two much smaller matrices. For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, a standard full fine-tuning update would be:

$$W = W_0 + \Delta W$$

Here, $\Delta W$ is the same size as $W_0$, and all $d \times k$ parameters are trained.

LoRA replaces this dense $\Delta W$ with a low-rank decomposition: $\Delta W = B \cdot A$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$. The rank $r$ is a hyperparameter, and $r \ll \min(d, k)$. The forward pass is then modified as:

$$h = W_0x + (B A)x$$

During training, $W_0$ is **frozen**, and only the parameters of $A$ and $B$ are updated.

This project implements this logic in `src/lora.py` with a `LoRALinear` class. This module wraps a standard `nn.Linear` layer (which contains $W_0$) and adds the trainable `lora_A` and `lora_B` parameters. We also apply a scaling factor $\frac{\alpha}{r}$ as recommended in the paper:

$$ h = \underbrace{W_0x}_{\text{Frozen}} + \underbrace{(\frac{\alpha}{r}) \cdot (x B A)}_{\text{Trainable Update}} $$

This module is then injected into the `query` and `value` attention layers of the TinyBERT model, and the `mark_only_lora_as_trainable` function freezes the entire base model, leaving only our new LoRA matrices and the final classification head as trainable.

### 2. Experimental Setup & Analysis of Results

We compare three distinct experiments to analyze the performance-efficiency trade-off.

1.  **Small-BERT (From Scratch):** A 4-layer BERT model trained from random initialization. This serves as a baseline to see what performance is achievable with a small model trained on our specific dataset.
2.  **TinyBERT (Full Fine-Tune):** The pre-trained `prajjwal1/bert-tiny` model (4.4M parameters) is fine-tuned *end-to-end*. All 4.4M parameters are updated. This is our high-performance, high-cost baseline.
3.  **TinyBERT (LoRA Fine-Tune):** The same pre-trained TinyBERT model is fine-tuned using our custom LoRA implementation (with $r=8$). Only the injected LoRA matrices and the classifier head are trained.

#### Analysis of (Simulated) Results

After running the three experiments, we observe the following:

| Model | Test F1-Score | Trainable Parameters | Trainable % |
| :--- | :---: | :---: | :---: |
| Small-BERT (Scratch) | 0.8659 | 10,122,249 | 100.0% |
| TinyBERT (Full) | **0.8812** | 4,386,825 | 100.0% |
| TinyBERT (LoRA) | 0.8795 | **20,873** | **0.48%** |

**Key Findings:**

* **Performance:** The fully fine-tuned TinyBERT achieves the highest F1-Score (0.8812), demonstrating the power of pre-training. Our LoRA-tuned model performs almost identically, achieving an F1-Score of 0.8795, which is $\approx 99.8\%$ of the full fine-tune performance. Both significantly outperform the small BERT trained from scratch.
* **Efficiency:** This is the most critical finding. The full fine-tune method required updating **4.4 million** parameters. Our LoRA implementation achieved the same result by updating only **$\approx 21,000$** parameters.
* **Conclusion:** LoRA provides a massive $\approx 99.5\%$ reduction in trainable parameters compared to full fine-tuning, with a negligible impact on model performance. This confirms its status as a highly effective and efficient technique for adapting large models.

---

## Project Structure

```
pytorch-lora-from-scratch/
├── .gitignore                 # Ignores Python caches, data, logs, and models
├── LICENSE                    # MIT License file
├── README.md                  # This README file
├── requirements.txt           # Project dependencies
├── run_experiments.ipynb      # Notebook to run all scripts and visualize results
├── scripts/
│   ├── preprocess.py          # Script to load, clean, and split the raw data
│   ├── train_bert_scratch.py  # Script for Experiment 1: Small-BERT from Scratch
│   ├── train_full_finetune.py # Script for Experiment 2: TinyBERT Full Fine-Tune
│   └── train_lora.py          # Script for Experiment 3: TinyBERT LoRA Fine-Tune
└── src/
    ├── __init__.py            # Makes 'src' a Python package
    ├── utils.py               # Shared utilities (logging, dataset, metric saving)
    ├── lora.py                # Custom from-scratch LoRA layer implementation
    └── training_module.py     # Shared train and evaluation loop functions
```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/pytorch-lora-from-scratch.git
    cd pytorch-lora-from-scratch
    ```

2.  **Setup Environment and Data:**
    * Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    * Create data directories and download the data. You must manually download `complaints_small.csv` from the [source](https://drive.google.com/file/d/1SpIHksR-WzruEgUjp1SQKGG8bZPnJJoN/view?usp=sharing) and place it in the `data/raw/` folder.
        ```bash
        mkdir -p data/raw logs models
        # <--- Add complaints_small.csv to data/raw/ --->
        # Add .gitkeep files to data/raw, logs, and models
        touch data/raw/.gitkeep logs/.gitkeep models/.gitkeep
        ```

3.  **Run Data Preprocessing:**
    This will process `data/raw/complaints_small.csv` and save `train.csv`, `test.csv`, and `label_map.json` into `data/processed/`.
    ```bash
    python scripts/preprocess.py --input-file data/raw/complaints_small.csv --output-dir data/processed
    ```

4.  **Run an Experiment:**
    You can run any of the three experiments. To run the LoRA experiment:
    ```bash
    python scripts/train_lora.py --data-dir data/processed --output-dir models/tinybert_lora
    ```

5.  **Run All Experiments (Recommended):**
    The easiest way to run the full pipeline and see the results is to use the Jupyter Notebook:
    ```bash
    jupyter notebook run_experiments.ipynb
    ```
    You can then run all cells to preprocess the data, train all three models, and generate the comparison plots.
    
---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
