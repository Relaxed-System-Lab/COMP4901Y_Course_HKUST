<div style="text-align:center">
<a href="https://hkust.edu.hk/"><img src="https://hkust.edu.hk/sites/default/files/images/UST_L3.svg" height="45"></a>


# COMP4901Y 2024 Spring 
</div>

**Lecturer**: [Binhang Yuan](https://binhangyuan.github.io/site/). 

**Teaching Assistant**: Ran Yan, Xinyang Huang.


## Overview

In recent years, foundation models have fundamentally revolutionized the state-of-the-art of artificial intelligence. Thus, the computation in the training or inference of the foundation model could be one of the most important workflows running on top of modern computer systems. This course unravels the secrets of the efficient deployment of such workflows from the system perspective. Specifically, we will i) explain how a modern machine learning system (i.e., PyTorch) works; ii) understand the performance bottleneck of machine learning computation over modern hardware (e.g., Nvidia GPUs); iii) discuss four main parallel strategies in foundation model training (data-, pipeline-, tensor model-, optimizer- parallelism); and iv) real-world deployment of foundation model including efficient inference and fine-tuning. 




## Syllabus 

| Date | Topic |
|-----|------|
|W1 - 01/31 | Introduction and Logistics [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%201%20-%20Introduction%20and%20Logistics.pdf)|
|W2 - 02/05, 02/07| Machine Learning Preliminary & PyTorch Tensors [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%202%20-%20Machine%20Learning%20Preliminary.pdf)|
|W3 - 02/14| Stochastic Gradient Descent [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%203%20-%20Stochastic%20Gradient%20Descent.pdf) |
|W4 - 02/19, 02/21 | Automatic Differentiation & PyTorch Autograd [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%204%20-%20Automatic%20Differentiation.pdf) |
|W5 - 02/26, 02/28 | Nvidia GPU Performance [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%205%20-%20Nvidia%20GPU%20Performance.pdf) & Collective Communication Library [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%206%20-%20Nvidia%20Collective%20Communication%20Library.pdf)|
|W6 - 03/04, 03/06| Transformer Architecture [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%207%20-%20Transformer%20Architecture.pdf) & Large Scale Pretrain Overview [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%208%20-%20Large%20Scale%20Pretrain%20Overview.pdf)|
|W7 - 03/11, 03/13| Data Parallel Training & Pipeline Parallel Training|
|W8 - 03/18, 03/20| Tensor Model Parallel Training & Optimizer Parallel Training|
|W9 – 03/25, 03/27| Mid-Term Review & Mid-Term Exam|
|W10 - 04/08, 04/10| Generative Inference Workflow & Hugging Face Library|
|W11 - 04/15, 04/17 | Generative Inference Optimization & Speculative Decoding |
|W12 - 04/22, 04/24 | Prompt Engineering Overview & Practices |
|W13 - 04/29 | Parameter Efficient Fine-tuning (LoRA)|
|W14 - 05/06, 05/08 | Guest Speech (TBD) & Final Exam Review |


## Grading Policy
- 4 Homework (4 $\times$ 5% $=$ 20%);
- Mid-term exam (30%);
- Final exam (50%).

## Homework 
| Topic | Release |   Due   |
|-------|---------|---------|
|[[Homework1]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/tree/main/Homework1)|2024-02-26 :heavy_check_mark:| 2024-03-18|
| Homework2 |2024-03-20| 2024-04-08|
| Homework3 |2024-04-10| 2024-04-24|
| Homework4 |2024-04-26| 2024-05-10|



