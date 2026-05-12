#The methods#

I have looked up some literature on hallucination detection adn tried out the methods that are told to be useful for the Qwen models.
The experiments outcome is presented in the table below.

1) The first experiment included change in aggregation function. T. Likun, H. Kuan-Wei, and W. Kevin [1] show that hallucination detection is stronger in later layers.
   T. Likun, H. Kuan-Wei, and W. Kevin [2] use layer-wise probing to detect hallucination. I wrote a separate script for evaluating the F1 score on a validation set for every layer (last token was used, because Qwen model concentrates all the semantics from previous tokens in the last token). The layer with the highest score (=6) became the layer I look on. So I wrote the aggregation function with the 6th layer. The outcome of the model is shown in the tabel (experiment 1).

2) The second experiment added geometrical features.
  For the geometrical features I used ICR-like norms of change between layers and Layer-wise semantic dynamics. Both ideas were taken from [1] and [3]. The function returns the vector of geometrical features. The outcome is represented in the table (experiment 2) - note that the experiment 2 also includes the steps in experiment 1 (and it is the same with every next experiment).

3) The third experiment concerned probe.py.
   I replaced one linear layer with a 2-layer MLP with Batchnorm, ReLU and Dropout. Also I added an internal train/validation split and early stopping. A. Urlana et al [4] show that even small MLP is more effective than linear layer. U. Köse and İ. Uysal [5] and M. Liu [6] state that it is important to have early stopping and dropout for small datasets to avoid overfitting. The outcome is shown in the table.
   
4) The last experiment concerned splitting. There are many researches (for ex. [1], [2]) that show that the sample from the same context must stay in one split. Otherwise it has bad influence on perfomance estimates. I replaced split with GroupShuffleSplit. It is done so that all the paraphrases go to training, validation or test but do not go to two or three groups of these at the same time.


| Название попытки | Accuracy (%) | F1 (%) | AUROC (%) |
| :--------------- | :----------: | :----: | :--------: |
| Majority-class baseline         |70.19        | 82.49   | NA       |
| Baseline Probe (test split)     | 73.08         | 82.72   | 73.18       |
| Experiment 1     | 73.08         | 81.58   | 73.64       |
| Experiment 2      | 71.15     | 81.71 | 73.71   |
| Experiment 3      | 73.08     | 81.33 | 73.49   |
| Experiment 4      | 70.19     | 82.49 | 73.71   |

#References#
[1]
T. Likun, H. Kuan-Wei, and W. Kevin, “InterpDetect: Interpretable Signals for Detecting Hallucinations in Retrieval-Augmented Generation,” Oct. 2025, doi: 10.48550/arxiv.2510.21538.
[2]
F. Thoresen and D. S. Smart, “A multilingual hallucination benchmark: MultiWikiQHalluA,” May 04, 2026. [Online]. Available: https://arxiv.org/abs/2605.02504v1
[3]
T. Likun, H. Kuan-Wei, and W. Kevin, “FRED: Financial Retrieval-Enhanced Detection and Editing of Hallucinations in Language Models,” Aug. 2025, doi: 10.48550/arxiv.2507.20930.
[4]
A. Urlana, G. Kanumolu, C. V. Kumar, B. M. Garlapati, and R. Mishra, “HalluCounter: Reference-free LLM Hallucination Detection in the Wild!,” arXiv.org, vol. abs/2503.04615, Mar. 2025, doi: 10.48550/arxiv.2503.04615.
[5]
U. Köse and İ. Uysal, “Persona Vectors in Controlling Hallucination of Small Large Language Models: A Safety-Oriented Analysis,” pp. 1–9, Oct. 2025, doi: 10.1109/cars67163.2025.11337402.
[6]
M. Liu, “A Unified Virtual Mixture-of-Experts Framework:Enhanced Inference and Hallucination Mitigation in Single-Model System,” Apr. 01, 2025. [Online]. Available: https://arxiv.org/abs/2504.03739v1

