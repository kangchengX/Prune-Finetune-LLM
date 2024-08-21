<p align="center">
    <h1 align="center">PRUNE-FINETUNE-LLM</h1>
</p>
<p align="center">
    <em>Refining Intelligence, Optimizing Performance Seamlessly</em>
</p>

<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg">
   <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg">
   <img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=default&logo=PyTorch&logoColor=white" alt="PyTorch">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
</p>

<br>
<details open>
  <summary>Table of Contents</summary><br>

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
- [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Usage](#usage)
- [Acknowledgments](#acknowledgments)
</details>
<hr>

##  Overview

Large Language Models (LLMs) have proven to be remarkably accurate and effective for several tasks such as summarisation, language translation, question answering and many others. However, to expand their capabilities and performance, these models have progressively increased in size. This growth has prompted research in two key areas: model compression and fine-tuning. Through compression techniques like pruning, redundant parameters and connections are trimmed, decreasing both memory usage and inference time. Fine-tuning then tailors the model's parameters to excel in designated domains or tasks, leveraging pre-trained natural language knowledge. This synergy optimises efficiency minimising impact on performance, addressing challenges of computational demands and task-specific proficiency. We seek to find the optimal ordering of this synergy, reporting our results on well-known LLM benchmarks. This study discusses a methodology for model compression and performance regeneration via Wanda pruning and LoRA fine-tuning. We investigate and quantify the impact on performance based on the ordering of pruning and fine-tuning for a compressed model on task-specific metrics, showing that _'Order Matters'_.

### Pipelines

<p align="center">
  <img src="plots\paper\procedure.jpg" alt="pipelines" width="300"/>
  <img src="plots\paper\procedure_i.jpg" alt="pipelines" width="340"/>
</p>

### Results

<table align="center">
  <tr>
    <td><img src="plots\bbh_comp_advanced.png" alt="bbh" width="310"/></td>
    <td><img src="plots\belebele_comp_advanced.png" alt="belebele" width="310"/></td>
  </tr>
  <tr>
    <td><img src="plots\mmlu_comp_advanced.png" alt="mmlu" width="310"/></td>
    <td><img src="plots\factoid_qa_comp_advanced.png" alt="factoid_qa" width="310"/></td>
  </tr>
</table>


---

##  Repository Structure

```sh
└── Prune-Finetune-LLM/
    ├── wanda
    ├── factoid_qa
    │   ├── __init__.py
    │   ├── freebase_qa.py
    │   └── FreebaseQA-eval.json
    ├── plots
    │   ├── [plot1].png
    │   ├── [plot2].png
    │   └── ...
    ├── main.py
    ├── eval.py
    ├── process.py
    ├── utils.py
    ├── constant.py
    ├── experiments.py
    ├── run.sh
    ├── plot_comparison_weights.py
    ├── plots_comparison_metrics.py
    ├── README.md
    └── requirements.txt
```

---

##  Modules

<details open><summary>.</summary>

| File                                 | Summary |
| ---                                  | --- |
| [main.py](main.py)                                         | Coordinates model operations, specifically managing the pruning, fine-tuning, and assessment of a large language model (LLM).  This servers as the main interface of executing model operation. |
| [eval.py](eval.py)                                         | Evaluates large language models (LLMs) across various datasets and metrics. |
| [process.py](process.py)                                   | Defines pruning and fine-tuning of the model. |
| [utils.py](utils.py)                                       | Serves as a utility module providing functions for model layer identification, language model response generation, response parsing, model loading, response validation, and results management. |
| [constant.py](constant.py)                                 | Defines critical pathways for various pipeline stages in the repository, and the path of python interpreter |
| [experiments.py](experiments.py)                           | Contains a series of different experiments, allowing for combinations of pruning and fine-tuning operations on pre-trained models under different pipelines, by executing `main.py`. |
| [plot_comparison_weights.py](plot_comparison_weights.py)   | Visualizes statistical distributions and differences of weights of LLMs, in order to compare different pipelines. |
| [plots_comparison_metrics.py](plots_comparison_metrics.py) | Generates comparative visualizations of performance metrics across different pruning and fine-tuning pipelines. |
| [requirements.txt](requirements.txt)                       | Depandencies for this repo. |
| [run.sh](run.sh)                                           | Executes multiple experiment pipelines, leveraging the `experiments.py` script. |


</details>

<details open><summary>wanda</summary>
This directory contains wanda pruning method, based on https://github.com/locuslab/wanda .

</details>

<details open><summary>factoid_qa</summary>
This directory contains factoid qa metric, which is a accuracy assessing model's ablity to store factual knowledge, based on https://github.com/kelvin-jiang/FreebaseQA .
</details>

<details open><summary>plots</summary>
This directory visualization results.
</details>

---

##  Getting Started

**System Requirements:**

* **Python**: `version 3.10.12`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the Prune-Finetune-LLM repository and submodules:
>
> ```console
> $ git clone --recurse-submodules https://github.com/kangchengX/Prune-Finetune-LLM.git
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd Prune-Finetune-LLM
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```

---

##  Acknowledgments

Thanks all 5 members of our team.

Alvaro, Fernandez; Aung, Htet; Carlos, Diez; Filippo, Fiocchi; Xu, Kangcheng

[**Return**](#overview)

---
