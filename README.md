 # Similar or Related: Spectral-Based Item Relationship Mining with Graph Convolutional Network for Complementary Recommendation
 
<font color='red'>The implementation code will be released after the acceptance of the paper.</font>

This is the PyTorch implementation for **SR-Rec** proposed in the paper **Similar or Related: Spectral-Based Item Relationship Mining with Graph Convolutional Network for Complementary Recommendation**.

> Gang-Feng Ma, Xu-Hua Yang, Haixia Long, and Yujiao Huang. 2024.

![img_1.png](img_1.png)

## 2. Running environment

We develop our codes in the following environment:

- python==3.9.18
- numpy==1.24.3
- torch==1.13.0
- torch-cuda=11.7

## 3. Datasets

| Dataset      | Appliances   | Grocery | Home |
| ------------ |----------| ------  | -------------|
| Items        | 804      | 38,548  | 75,514       |
| Edges        | 8,290    | 642,884 | 776,766      |
| Avg. Degree  | 20.6     | 33.4    | 20.6         |

## 4. How to run the codes


```python9
 python run.py
```
