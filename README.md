# Predictive Process Monitoring

The increasing amount of data together with technological improvements lead to significant changes in the strategies of machine maintenance. The possibility of monitoring machine’s conditions has arisen the Predictive Maintenance (PM). PM had evolved in the last decade and is characterized by the use of the machine’s historical time series data, collected through sensors. Using the available data, it’s possible to provide effective solutions with Machine Learning and Deep Learning approaches. Predictive Maintenance allows to minimize downtime and maximize equipment lifetime.

I’ll use a popular dataset available in NASA’s data repository, called PHM08. It’s a collection of data introduced for the first time in 2008 for a challenge competition at the first conference on Prognostics and Health Management. It’s a multivariate time series, that contains 218 turbofan engines, where each engine data has measurements from 21 sensors. Each engine starts to operate normally and ends in failure. The goal is to predict the RUL of the components. Below, there are the steps that will be followed to build the predictive maintenance algorithm. If you have already created a machine learning model with different datasets in the past, you’ll observe that the only difference is that you need an additional task. You have to compute the Remaining Useful Life values, that are needed to be compared with the predictions.

## Article related to the code

[Predicting the Remaining Useful Life of Turbofan Engine](https://pub.towardsai.net/predicting-the-remaining-useful-life-of-turbofan-engine-f38a17391cac?sk=32f6f00a3f24c7fb4bf6082c51ba63ff)
