# ATTD-2024-Poster-Presentation

*TEMPORAL FEATURE-DRIVEN TECHNIQUE FOR PRECISION MEDICINE: EARLY IDENTIFICATION OF INDIVIDUALS AT HIGH RISK FOR STAGE 3 TYPE 1 DIABETES*

## Abstract

**Background and aims**: Early identification of individuals at high risk of developing type 1 diabetes (T1D) is essential for preventing or delaying the clinical onset of the disease. We propose a novel feature-driven technique to identify high-risk populations for T1D. The technique consists of: (i) observing autoantibody persistence as a temporal phenomenon. (ii) a risk score that accounts for the variability in autoantibody persistence, and (iii) survival models combining (i) and (ii). 

**Methods**: The proposed technique is based on the redefinition of autoantibody persistence as a temporal phenomenon and the development of a probabilistic risk score that accounts for the variability in autoantibody persistence. The technique is evaluated on a sub-dataset of the TEDDY study population, which consists of children aged between 365 days and 5159 days. Linear (Cox Proportional Hazard (CPH)) and non-linear (Decision Tree (DT), and Random Forest (RF)) survival models that use the probabilistic risk score and engineered features, such as genetic, demographic, and physical features, are presented. Their performances are evaluated using time-dependent AUC (AUC(t)) and Concordance index with inverse probability of censoring weight (C-ipcw). 

**Results**: The results demonstrate the effectiveness of the proposed technique and engineered features in differentiating high- and low-risk populations. For the respective models: C-ipcv: RF = 0.98, DT = 0.97, CPH = 0.97, and AUC(t): RF = 0.995, DT = 0.992, CPH = 0.991.

**Conclusion**: We propose a novel feature-driven technique to identify high-risk populations for T1D clinical diagnosis. The presented models can be used for T1D screening and early detection.

## Dataset

- [TEDDY-Study](https://repository.niddk.nih.gov/studies/teddy/?query=type%201%20diabetes)
- [Config-files](https://github.com/S-B-Iqbal/ATTD-2024-Poster-Presentation/blob/main/code/src/conf/config.yaml) contains the list of utilized datasets
- Data-dictionary is available under "Study Documents"

## Code

- `create_processed_df.py` estimates the relevant risk scores for each individual under the column `SCORE_<AB>_MEMORY` for each auto-antibody
- `train_survival_models.py` trains the models

