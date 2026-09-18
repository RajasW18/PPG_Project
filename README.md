# PPG_Project
This is a repository made for maintenance of our Mini Project, group members are - Rajas Wagle, Dhruv Warik, and Vivaan Tasker.
Project Guide is Dr. Prashant V Kasambe.

## Project Overview

This project aims to analyze Photoplethysmography (PPG) data to assess the risk of Myocardial Infarction. 

**Dataset:**
We utilized the [Photoplethysmography (PPG) dataset from the UCI Machine Learning Repository](https://www.kaggle.com/datasets/ucimachinelearning/photoplethysmography-ppg-dataset?resource=download).

**Data Processing Pipeline:**
1. **Feature Extraction:** Raw PPG data was initially processed using our `ppg_featureExtractor.py` script. This generated an intermediate dataset containing a wide range of extracted features (`PPG_Extracted_Features.csv`).
2. **Feature Analysis & Reduction:** We conducted an in-depth analysis to determine the relative importance of these extracted features.
3. **Final Dataset:** Based on the feature importance analysis, we reduced the dataset to the 5 most critical input features and 1 output label. The output label classifies the data into two categories:
   - `0`: Normal
   - `1`: High risk for Myocardial Infarction
   
This streamlined dataset is saved as `PPG_Dataset6Features.csv` and serves as the foundation for our predictive modeling.

## Directory Structure

Here is a brief overview of the key files and directories in this repository:

- **Data Processing & Datasets:**
  - `ppg_featureExtractor.py`: Script used to extract initial features from raw PPG data.
  - `PPG_Extracted_Features.csv`: Intermediate dataset containing all initially extracted features.
  - `PPG_Dataset6Features.csv`: Final reduced dataset used for our modeling.

- **Hardware Acceleration (HLS & Vivado):**
  - `ppg_ConversiontoHLS.py`: Script used to convert our Python machine learning model into High-Level Synthesis (HLS) C/C++ code.
  - `ppg_rf_hls_12bit/`: The final built HLS project for hardware implementation.
  - `Vivado Files/vivado_code/`: Contains the core code for the Vivado hardware design.
  - `Vivado Files/Updated Project/`: Contains the updated project files for Vivado.

- **Web Development / Front-end:**
  - `Vivado Files/Prakalp/`: Contains the files for the web development aspect of the project.
