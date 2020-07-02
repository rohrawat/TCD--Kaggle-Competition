# TCD--Kaggle-Competition
This competition was organized as a coursework for Machine Learning module.
Dataset was provided to predict salary from various features.
Data was untidy and it has a huge number of categorical features. To deal with this, Target encoding was used to replace categorical features. Missing values were imputed accordingly using mean and median instead of dropping it.
CatBoostRegressor, XGBost, Neural Network was used to predict salary, but CatBoostRegressor gave a good result.
Two variant of CatBoostregresor was used to predict salary based on learning rate. Later mean of both the result was taken to improve the overall score.
