Implementation of paper: CrossGCL: Cross-view Graph Contrastive Learning with Dual Task for Drug Recommendation

### How to run the code

1. Files Prepare

   1. download MIMIC Dataset files `DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv PROCEDURES_ICD.csv` and put them on the folder `datasets/mimic/mimic`

      ps: We have to follow the MIMIC Dataset policy and not share the datasets. Practioners can visit https://physionet.org/content/mimiciii/1.4/ and apply for the permission to access the MIMIC-III dataset

   2. files from external sources `drug-atc.csv, drug-DDI.csv, ndc2atc_level4.csv, ndc2rxnorm_mapping.txt, idx2drug.pkl`

      I have put all these files in the `datasets/mimic` folder, except for the `drug-DDI.csv`, which needs to be downloaded from [Google Drive](https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing).

2. Data Process

   (For a fair comparision, we use the same data and pre-processing scripts used in [Safedrug](https://github.com/ycq091044/SafeDrug) and [COGNET](https://github.com/BarryRun/COGNet))
   1. processing the data

      > ```bash
      > python processing.py
      > ```

   2. graph construct

      > ```bash
      > python graph_construct.py
      > ```
      >

4. model train & infer

   1. train graph representation learning module
      > ```bash
      > python train.py --mode=train
      > ```
      >
   
   2. train drug size predict module and infer
      > ```bash
      > python train.py --mode=infer
      > ```

