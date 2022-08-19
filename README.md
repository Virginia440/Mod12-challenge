# Mod12-challenge
# UW-finctech-2022
This is  a public repo for the Module 12 Challenge of the UW Fintech Bootcamp in 2022.


## Technologies and Libraries

Jupyter lab
pandas. 1.3.5
hvplot 0.8.0
scikit-learn 1.0.2


## Installation Guide

Install jupyter lab by running the command jupyter lab in your terminal

Install the following dependencies an dmocdules from the libraries above

```
  Import hvplots.pandas
  Import panda as pd
  From pathlib import Path
  from sklearn.metrics import balanced_accuracy_score
  from sklearn.metrics import confusion_matrix
  from imblearn.metrics import classification_report_imbalanced

```


## Overview of the analysis

* Purpose of the analysis
Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. This project follows the model-fit-predict-evaluate pattern. It employs various techniques to train and evaluate models with imbalanced classes. It makes use of a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. 

* Financial Information on the data.
The dataset used has 75036 healthy loans and 2500 unhealthy loans. A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting. The data is separated into two DataFrames: y dataframe(label data), represented y the 'loan status' column and and X dataframe(feature data) representing all other columns in the dataset but the loan status column.

* Description of the stages of the machine learning process
1. Read in the CSV file from the Resources folder into a Pandas DataFrame
2. Create a Series named y that contains the data from the "Default" column of the original DataFrame which is the "loan_status" columnn. Note that this Series will contain the labels. Create a new DataFrame named X that contains the remaining columns from the original DataFrame
3. Split the label and feature datasets into testing and training sets using the (train_test_split) function.
4. Check the magnitude of imbalance in the data set by viewing the number of distinct values (value_counts) for the labels
5. Resample the training data by using RandomOverSampler
6. Check the number of distinct values (value_counts) for the resampled labels.
7. Fit a LogisticsRegeression model using the fit function.
8.  Using the logistic regression model, predict the values for the  sets by calling the (predict) function on the training and testing sets. This involves making predictions for both the training features (X_train) set and the testing features (X_test) set.
9. Print the confusion matrixes, accuracy scores, and classification reports for the datasets.
10. Repeat steps 7 through 9 using both original data and resampled data.


## Results
* Machine Learning Model 1:(Original dataset)
  * Description of Model 1 Accuracy, Precision, and Recall scores.

    The precision is on what could referred to as the perfect mark for the 0 class (1.00), but lower for the 1 class (0.85). Recall that 0 represents healthy loans. And, 1 represents high risk loans. Therefore, the model does better in predicting healthy loans. For instance, say for every 50 predictions that a loan will be healthy, 100%  of the results were correct (0.50 × 100). However, this is overly accurate and could be due to the imbalanced classes. The size of the healthy loans exceeds that of the high risk loans.

    The recall for the 0 and 1 classes has a good balance(0.99 and 0.91).The testing dataset had 619 actual 1 values. This is found in the number in the “sup”, which stands for “support”, column in the preceding classification report. This means that this model correctly predicted 563 of them (0.91 × 619). By accurately identifying 91% of all high risk loans, the model did a fairly good job.
 
    The balanced accuracy score is higher for the resampled data (0.99 vs 0.95), meaning that the model using resampled data was much better at detecting true positives and true negatives.  

* Machine Learning Model 2:(oversampled dataset)
  * Description of Model 2 Accuracy, Precision, and Recall scores.
    The precision is on what could referred to as the perfect mark for the 0 class (1.00), but lower for the 1 class (0.84). Recall that 0 represents healthy loans. And, 1 represents high risk loans. The model does an almost similar prediction as that that uses original data and with a precision of (0.85) versus the model that used the oversampled data produced a precision of (0.84). So, the logistic regression model that used the original imbalanced data did just slightly better by (0.01) in making predictions for the 1 class.

    The recall for the 0 and 1 classes has a good balance(0.99 and 0.99). However, compared to the model that used original data which had a recall of (0.91), the model that used the oversampled data was dramatically more accurate at predicting the high risk loans. 

    The balanced accuracy score is higher for the resampled data (0.99 vs 0.95), meaning that the model using resampled data was much better at detecting true positives and true negatives. 

---

## Summary
Both models have almost similar predictions. However, the model using resampled data was much better at detecting high risk loans than the model generated using the original, imbalanced dataset.
## Contributors


## License
 The code is made without a license, however, the materials used for research are licensed.
---


