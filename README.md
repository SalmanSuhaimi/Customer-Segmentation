# Customer-Segmentation

## Project Description
Customer segmentation involves categorizing customers based on common characteristics such as age, gender, profession, and interests. This helps a company understand customer needs and tailor marketing strategies for more effective targeting. A bank facing declining revenue discovered a decrease in client deposits as the main issue. To address this, the bank plans marketing campaigns to encourage more deposits. The campaign aims to gather customer feedback and satisfaction through key elements like customer segmentation and promotional strategies. A well-identified strategy is crucial for the bank to boost revenue and growth.

## Problem Statement:
- The bank is experiencing a decline in revenue attributed to a reduction in client deposits. 
- To address this issue, the bank aims to implement marketing campaigns focused on encouraging more deposits.
- The problem is rooted in the need to identify effective strategies within the campaign, including customer segmentation and promotional tactics, to understand and meet customer needs and satisfaction, ultimately revitalizing revenue and fostering growth for the bank.

## Objective:
- The objective of this project is to leverage a deep learning approach using TensorFlow, specifically incorporating Dense, Dropout, and Batch Normalization layers,to develop a predictive model for the outcome of marketing campaigns conducted via phone.
- The primary goals are to achieve an accuracy rate exceeding 70%, visualize the training loss and accuracy on TensorBoard, and enhance the efficiency of the training and testing processes by creating modularized functions in the form of classes for repeated tasks.
- The ultimate aim is to deliver a robust model that can assist in optimizing marketing strategies, particularly in persuading clients to deposit money into the bank, thereby addressing the observed decline in revenue and fostering potential growth for the bank.

### Coding Flow
1. Importing Packages: The code begins by importing necessary libraries and packages such as TensorFlow, Pandas, NumPy, and Matplotlib.

2. Loading Data: The dataset is loaded from a CSV file ('train.csv') into a Pandas DataFrame.

3. Data Inspection and Cleaning:

  - The structure of the dataset is examined using methods like keys(), info(), and describe().
  - Unnecessary columns (like 'id') are dropped.
  - Histograms and other visualizations are used to explore the distribution of data.
  - Missing values are handled by filling or dropping them.
  - Numerical and categorical columns are identified.

4. Data Preprocessing:

  - Ordinal encoding is applied to certain categorical columns.
  - Features (X) and target variable (y) are separated.
  - One-hot encoding is applied to the target variable (y) to convert it into a format suitable for training.

5. Train-Test Split:

  - The dataset is split into training and testing sets.

6. Model Building:

  - A Sequential model is created using TensorFlow, consisting of Dense, Batch Normalization, and Dropout layers.
  - The model is compiled with the Adam optimizer and categorical cross-entropy loss.
  - Early stopping and TensorBoard callbacks are set up.
![epoch_accuracy]()
<p align="center">
<img src="https://github.com/SalmanSuhaimi/Customer-Segmentation/assets/148429853/fd07708f-f46c-4a46-9384-4a2ae10be0a4"/>
<p>
The results from tensorboard shows the epoch accuracy is consistent 85% above.
  
7. Training the Model:

  - The model is trained on the training set, and the training process is monitored using callbacks.

8. Model Evaluation:

  - The model is evaluated on the test set, and the loss and accuracy are printed.
<p align="center">
<img src="https://github.com/SalmanSuhaimi/Concrete-Crack-Images-for-Classification/assets/148429853/15db0669-3bf1-42da-843f-7b65c3db9fe5"/>
<p>
The picture shows the accuracy is 89.83% and loss is 0.2408

9. Model Saving:

  - The trained model is saved in the 'models' folder.
