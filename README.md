<div style="text-align:center">
<a href="https://hkust.edu.hk/"><img src="https://hkust.edu.hk/sites/default/files/images/UST_L3.svg" height="45"></a>


# COMP4901Y 2024 Spring

</div>

<h2 style="text-align: center;"> Large-Scale Machine Learning for Foundation Models </h2>

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
|W7 - 03/11, 03/13| Data Parallel Training [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%209%20-%20Data%20Parallel%20Training.pdf) & Pipeline Parallel Training [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%2010%20-%20Pipeline%20Parallel%20Training.pdf)|
|W8 - 03/18, 03/20| Tensor Model Parallel Training [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%2011%20-%20Tensor%20Model%20Parallel%20Training.pdf) & Optimizer Parallel Training [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%2012%20-%20Optimizer%20Parallel%20Training.pdf)|
|W9 – 03/25, 03/27| Mid-Term Review [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%2013%20-%20Midterm%20Review.pdf) & Mid-Term Exam :heavy_check_mark:|
|W10 - 04/08, 04/10| Generative Inference [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%2014%20-%20Generative%20Inference.pdf) & Hugging Face Library [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%2015%20-%20Hugging%20Face%20Transformers%20Inference%20API.pdf)|
|W11 - 04/15, 04/17 | Generative Inference Optimization [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%2016%20-%20Generative%20Inference%20Optimization.pdf) & Speculative Decoding [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%2017%20-%20Speculative%20Decoding.pdf)|
|W12 - 04/22, 04/24 | Prompt Engineering [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%2018%20-%20Prompt%20Engineering.pdf) |
|W13 - 04/29 | Parameter Efficient Fine-Tuning [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%2019%20-%20Parameter%20Efficient%20Fine-Tuning.pdf) |
|W14 - 05/06, 05/08 | Final Exam Review [[Slides]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/blob/main/Slides/Lecture%2020%20-%20Final%20Review.pdf)|


## Grading Policy
- 4 Homework (4 $\times$ 5% $=$ 20%);
- Mid-term exam (30%);
- Final exam (50%).

## Homework 
| Topic | Release |   Due   |
|-------|---------|---------|
|[[Homework1]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/tree/main/Homework1)|2024-02-26 :heavy_check_mark:| 2024-03-18 :heavy_check_mark:|
|[[Homework2]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/tree/main/Homework2)|2024-03-20 :heavy_check_mark:| 2024-04-08 :heavy_check_mark:|
|[[Homework3]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/tree/main/Homework3)|2024-04-11 :heavy_check_mark:| 2024-04-25 :heavy_check_mark:|
|[[Homework4]](https://github.com/Relaxed-System-Lab/COMP4901Y_Course_HKUST/tree/main/Homework4)|2024-04-26 :heavy_check_mark:| 2024-05-10 :heavy_check_mark:|



