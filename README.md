# CATS (NeurIPS 2024)

#### This repository is an official PyTorch implementation of CATS: [Are Self-Attentions Effective for Time Series Forecasting?](https://arxiv.org/abs/2405.16877)

## Key Design of CATS

![alt text](https://github.com/dongbeank/CATS/blob/main/pic/CATS_architecture.png)

#### ⚡ Cross-Attention Only Time Series transformer

CATS removes self-attention and retains only cross-attention in its transformer architecture. This design choice aims to better preserve temporal information in time series forecasting, addressing the potential loss of such information during the embedding process in traditional transformer models.

#### ⚡ Time and Memory Efficiency

CATS achieves improved time and memory efficiency compared to traditional self-attention-based transformers. While self-attention complexity grows quadratically with input length $$(O(L^2))$$, CATS' cross-attention-only approach scales linearly $$(O(LT))$$.

#### ⚡ Enhanced Parameter Sharing

CATS implements extensive parameter sharing across all layers and dimensions for each horizon-dependent query. This approach, including shared projection layers, significantly reduces parameter count and improves computational efficiency in both training and inference phases.


## Efficiency of CATS

We conducted extensive experiments to compare CATS with other state-of-the-art models for long input sequences. Our results demonstrate that CATS outperforms existing models in both efficiency and effectiveness.

### Performance Across Various Input Lengths

CATS maintains robust performance as input length increases, unlike some complex models that suffer from increased computational burdens.

![Efficiency Comparison Table](https://github.com/dongbeank/CATS/blob/main/pic/Efficiency_table.png)

### Handling Significantly Longer Sequences

We pushed CATS further by testing it with significantly longer input sequences (2880 time steps) and compared it to other models using shorter inputs (512 time steps). The results were remarkable:

- CATS demonstrated better efficiency in terms of parameters, running time, and memory usage, even when processing nearly 5 times more data.
- It achieved this while maintaining superior forecasting performance.

![Efficiency Comparison Graph](https://github.com/dongbeank/CATS/blob/main/pic/Efficiency_graph.png)

## Understanding Periodic Patterns with Cross-Attention

To better understand how CATS processes time series data, we visualized its cross-attention mechanisms. We used a simple time series composed of two independent signals with different periodicities (where $\tau = 24$, $S = 8$, and $k = 5$).

<p align="center">
<img src="https://github.com/dongbeank/CATS/blob/main/pic/Equations.png" width="60%">
</p>

<p align="center">
<img src="https://github.com/dongbeank/CATS/blob/main/pic/Cross_attention_score_map.png" width="80%">
</p>

These maps reveal CATS' ability to capture both shocks and periodicities in the signal:
- The left score map shows higher attention scores for patches containing shocks in the same direction.
- The right score map clearly demonstrates the correlation over 24 steps, reflecting the model's capture of signal periodicity.

This visualization confirms CATS' effectiveness in leveraging periodic information for accurate predictions.


## Forecasting Results

CATS demonstrates superior performance across most datasets and forecasting horizons. CATS shows competitive results, often achieving the best or second-best scores in various time series forecasting tasks.

<img src="https://github.com/dongbeank/CATS/blob/main/pic/Experimental_results.png">

## Getting Started

### Requirements
To set up the environment, follow these steps:
1. Install Python 3.9
2. Install the required packages:
```
pip install -r requirements.txt
```
### Data Preparation
To replicate the experiments in our [paper](https://arxiv.org/abs/2405.16877), follow these steps:

1. Download the dataset from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).
2. Create a folder named `./dataset` in the root directory of this project.
3. Place all downloaded files and folders within the `./dataset` folder.

### Scripts
We provide various scripts for different datasets and input lengths. Here are a couple of examples:

1. For the ETTm1 dataset with 512 input length:
```bash
bash ./scripts/ETTm1_512_input.sh
```
2. Specifically, for the Traffic dataset with large input (2880):
```bash
bash ./scripts/Traffic_2880_Large_input.sh
```
You can find more scripts in the `./scripts` folder for other datasets and input lengths.

## Citation
If you find this repo useful for your research, please cite our paper:
```bibtex
@inproceedings{kim2024self,
  title={Are Self-Attentions Effective for Time Series Forecasting?},
  author={Kim, Dongbin and Park, Jinseong and Lee, Jaewook and Kim, Hoki},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  year={2024}
}
```

## Acknowledgements
We would like to express our appreciation for the following GitHub repositories, which provided valuable code bases and datasets:

- [Autoformer](https://github.com/thuml/Autoformer)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [PatchTST](https://github.com/yuqinie98/PatchTST)

## Contact
If you have any questions or want to use code, please contact dongbin413@snu.ac.kr
