 # Similar or Related: Spectral-Based Item Relationship Mining with Graph Convolutional Network for Complementary Recommendation
 
This is the PyTorch implementation for **SR-Rec** proposed in the paper **Similar or Related: Spectral-Based Item Relationship Mining with Graph Convolutional Network for Complementary Recommendation**.

>  

![img_1.png](img_1.png)

## 1. Running environment

We develop our codes in the following environment:

- python==3.9.18
- numpy==1.24.3
- torch==1.13.0
- torch-cuda=11.7

## 2. Datasets

| Dataset      | Appliances   | Grocery | Home |
| ------------ |----------| ------  | -------------|
| Items        | 804      | 38,548  | 75,514       |
| Edges        | 8,290    | 642,884 | 776,766      |
| Avg. Degree  | 20.6     | 33.4    | 20.6         |
|Avg. Clustering Coefficient| 0.373 |0.370 | 0.294 |
| Sparsity     | 97.432%  | 99.913% | 99.973%      |
## 3. Data Preprocessing
way1:
- Download raw_data from https://nijianmo.github.io/amazon/index.html.
- Put the meta data file in <tt>./data_preprocess/raw_data/</tt>.
- Set the dataset name (i.e., <tt>$dataset</tt>) in run.sh, run preprocessing by 
    ```
    cd data_preprocess
    sh run.sh
    ```

way2:
- We provide the download method of [Google Cloud](https://drive.google.com/drive/folders/1kCx6WllSrI9KUVCdo2u2BoAmuSnlHYPm?usp=sharing) for the convenience of use.


Taking the Appliances dataset as an example, the correspondence between files and folders is as follows


    data_preprocess/
    │
    ├── data/
    │   └── Appliances.json
    │
    ├── embs/
    │   └── Appliances_embeddings.npz
    │
    ├── processed/
    │   └── Appliances.npz
    │
    ├── raw_data/
    │   └── meta_Appliances.json
    │
    ├── stats/
    │
    ├── tmp/
        ├── Appliances_cid2_dict.txt
        ├── Appliances_cid3_dict.txt
        ├── Appliances_cor.edges
        ├── Appliances_id_dict.txt
        ├── Appliances_sim.edges
        ├── filtered_Appliances_cor.edges
        ├── filtered_Appliances_sim.edges
        └── filtered_meta_Appliances.json




## 4. How to run the codes


```python9
 python run.py
```
The structure of this code is based on [SComGNN](https://github.com/luohaitong/SComGNN). Thanks for their excellent work!
