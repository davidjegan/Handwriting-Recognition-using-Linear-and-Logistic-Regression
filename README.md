# Handwriting-Recognition-using-Linear-and-Logistic-Regression
This code addresses how we could identify the similarity between two handwritten samples using Linear regression, Logistic regression and Tensorflow methodologies. If the algorithm detects a match, then 1 is returned and similarily  0 is returned if there is dissimilarity between the two images. Though the results are scalar and discrete (0,1), we consider linear/logistic regression and tensorflow to obtain continuous value rather than a discrete on, to find the percent of match. 


## Approach
 There are four stages in this problem
1. Create training data
2. Linear regression
3. Logistic regression
4. Neural Networks using Tensorflow


### Methodology
#### Data Preparation
For the algorithm to learn we provide a training dataset. This data is crucial for the prediction as the
model considers this data as Ground truth and tries to learn. The dataset here uses the image snippets of
AND extracted from CEDAR Letter dataset. There are two dataset types, namely Human observed and
Stochastic Gradient Concavity.

##### Human Observed dataset
The Human Observed dataset contains the details extracted from the image snippet by a human. Thus there
are 9 features for a image, and for a pair of images there are 18 samples. There are 791 pairs of same writer
and 293,032 pairs of different writer extracted from the image database.

##### Gradient Structural Concavity dataset
The Gradient Structural Concavity dataset contains the details extracted from the image snippet by a
program. Thus there are 512 features for a image, and for a pair of images there are 1024 samples. There
are 71,531 pairs of same writer and 762,557 pairs of different writer extracted from the image database.
##### Feature Data
As observed, the number of different writer pairs dominate, so in order to avoid model biasing we consider
the same number of same/different writer records in both the samples, 1582 in case of Human observed
dataset and 10000 in case of SGC dataset. The feature dataset concatenated with the target is shuffled to
facilitate random training and test sample sets, so that the model is more robust.
##### Feature Concatenation
For regression, the more the data features, the more is the chances a system can learn about the model. So
thus we add the ’n’ features for the image pairs and pass it as ’2n’ features. Thus human and SGC has 18
and 1024 features respectively.
##### Feature Subtraction
To minimize the time and space complexity, we subtract the two ’n’ features for the image pairs and pass it
as ’n’ features. Thus human and SGC has 9 and 512 features respectively.
##### Target Data
The actual result serves to test our model behaviour. Thus we divide the target values into percents of actual
data called training, validation and testing result set, in such a way that the sets do not overlap. These
values are then compared with the obtained results to compare the accuracy of our model.


##### Thus there are 4 different datasets in our paper.
1. Human Observed dataset with feature concatenation
2. Human Observed dataset with feature subtraction
3. SGC dataset with feature concatenation
4. SGC dataset with feature subtraction
