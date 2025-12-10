# GCN-OC
# Multimodal Data-Based Graph Convolutional Networks for Predicting Outcomes in Patients with Ovarian Cancer Receiving Neoadjuvant Chemotherapy: A Multicenter Retrospective Study

## Project Structure

### Step 1. EV_GCN-OC-OA 
- Constructs population graph based on the patients' baseline non-imaging characteristics
- Utilizes baseline features to build patient similarity graphs before neoadjuvant chemotherapy
- PAE_main.py is used to build patient similarity estimates, and the generated PAE models are stored in the save_models folder

### Step 2. GCN_HGNN_OC.py
- Implements DHGN prognostic model construction, training and external test
- Use the "edge_weights_..._pred_labels" output from Step 1 as the patient similarity estimate for the dataset
- Generates prognostic scores for survival outcome prediction in ovarian cancer patients receiving neoadjuvant chemotherapy

## Important Note
Before running the project:
- Update the cut-off value in the code according to your own dataset (marked with comments)
  
Materials for readers:
- Laboratory_CT_Features.csv: Clinical features (78 features) and CT features of 20 patients in the training dataset

## Acknowledgments
We gratefully acknowledge the contributions of Huang, Yongxiang and Chung, Albert CS to the PAE model, and Gao, Yue and Feng, Yifan and Ji, Shuyi and Ji, Rongrong to the GCN model.
