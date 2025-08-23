# CheXSBT

Code for Master's Thesis- Transform(AI)ng Radiology with CheXSBT: Integrating Dual-Attention Swin Transformer with BERT for Seamless Chest X-Ray Report Generation.

Submitted in accordance with the requirements for the degree of MSc Advanced Computer Science at the University of Leeds, UK.

This work was presented at the 2025 Medical Image Understanding and Analysis (MIUA) Conference and published in Springer Lecture Notes in Computer Science (LNCS), vol 15916.

Cite: Khandeparker, A., Lu, P. (2026). Transform(AI)ng Radiology with CheXSBT: Integrating Dual-Attention Swin Transformer with BERT for Seamless Chest X-Ray Report Generation. In: Ali, S., Hogg, D.C., Peckham, M. (eds) Medical Image Understanding and Analysis. MIUA 2025. Lecture Notes in Computer Science, vol 15916. Springer, Cham. https://doi.org/10.1007/978-3-031-98688-8_12


Download the MIMIC-CXR dataset from https://physionet.org/content/mimic-cxr/2.1.0/.


To run the code:

1. Create a directory `dataset` and move all the files from MIMIC-CXR dataset to this directory.

2. Run `preprocessing.py` to process the text reports from the dataset. This will create 3 files in the  `dataset` directory, namely `train.csv`, `val.csv` and `test.csv`.

3. Run `main.py` to train the model. The trained model will be saved in `models` directory (Please create this directory before training the model for the first time)

4. Run `test.py` to test the trained model.



This work was undertaken on the Aire HPC system at the University of Leeds, UK.
