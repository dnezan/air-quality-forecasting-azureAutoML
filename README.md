![banner](https://raw.githubusercontent.com/dnezan/air-quality-forecasting-azureAutoML/master/img/repo_banner.png?token=GHSAT0AAAAAAB7WFMCSPTHSGWXNAXQXTXWOZBNLS3Q)
# Air Quality Forecasting using Azure AutoML in Python

## Overview
 
This project focuses on using Azure AutoML and Python SDK to build a time series forecasting model for air quality in my home city, Chennai. By leveraging AutoML's automated machine learning capabilities and the flexibility of Python SDK, the model aims to predict future air quality with high accuracy, enabling government agencies and individuals to make data-driven decisions to combat air pollution.

The research dataset was sourced from [Open Government Data (OGD) Platform India - Tamil Nadu](https://tn.data.gov.in/catalog/historical-daily-ambient-air-quality-data) and underwent cleaning and preparation after being downloaded.

AutoML can be used for the following tasks  
- Classification 
- Regression
- **Time series forecasting**
- Computer vision
- Natural language processing

Because our problem statement is regarding time series forecasting, we can use Azure AutoML to iterate over a number of algorithms to find the best fit. By default, Azure AutoML deploys the following algorithms.

|                     |                             |                      |                 |
|---------------------|-----------------------------|----------------------|-----------------|
| AutoARIMA           | Decision Tree               | ForecastTCN          | Naive           |
| Prophet             | Arimax                      | Gradient Boosting    | SeasonalAverage |
| Elastic Net         | LARS Lasso                  | ExponentialSmoothing |                 |
| Light GBM           | Extremely Randomized Trees* | SeasonalNaive        |                 |
| K Nearest Neighbors | Random Forest               | Average              |                 |

Since this is a proof of concept, we will compare algorithm performance between **AutoARIMA**, **Prophet**, **Arimax**, and **ForecastTCN** (Deep Neural Network).

## Environment Variables

To run this project in Azure AutoML, you will need to add the following environment variables to your config.json file

`subscription_id`

`resource_group`

`workspace-name`

## Dataset

Throughtout the years, the number of monitoring stations in Tamil Nadu has increased, as well as their capabilities. The very early years of 1987 had very few agencies that were only able to detect a few types of pollutants. In 2015, we now have data on SO2, NO2, RSPM/PM10, and SPM levels across 11 different monitoring agencies. 

| Sampling Date | Location of Monitoring Station | SO2 | NO2 | RSPM/PM10 | SPM | PM2.5 |
|---------------|--------------------------------|-----|-----|-----------|-----|-------|
|               |                                |     |     |           |     |       |
|               |                                |     |     |           |     |       |

To choose a high quality subset of data, we want to choose from locations that have consistent readings for pollutants in recent years. 

| Location of Monitoring Station                   | 2010  | 2011  | 2012  | 2013  | 2014  | 2015  |
|--------------------------------------------------|-------|-------|-------|-------|-------|-------|
| **Kathivakkam, Municipal Kalyana Mandapam, Chennai** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |
| **Govt. High School, Manali, Chennai.**              | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |
| **Thiruvottiyur, Chennai**                           | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |
| Madras Medical College, Chennai                  | **Y** | **Y** | **Y** | N     | **Y** | **Y** |
| NEERI, CSIR Campus Chennai                       | **Y** | **Y** | **Y** | N     | **Y** | **Y** |
| Thiruvottiyur Municipal Office, Chennai          | **Y** | **Y** | **Y** | N     | **Y** | **Y** |
| Adyar, Chennai                                   | N     | N     | N     | **Y** | **Y** | **Y** |
| Anna Nagar, Chennai                              | N     | N     | N     | **Y** | **Y** | **Y** |
| Thiyagaraya Nagar, Chennai                       | N     | N     | N     | **Y** | **Y** | **Y** |
| Kilpauk, Chennai                                 | N     | N     | N     | **Y** | **Y** | **Y** |
| Vallalar Nagar, Chennai                          | N     | N     | N     | N     | N     | **Y** |

We will be forecasting air pollution in three different areas due to the availability of data (Kathivakkam, Manali, and Thiruvottiyur) in the years 2010-2015. Coincidentally, all three areas fall under Industrial locations and will therefore see pollutant levels that are higher than that of the surrounding residential areas.

## Data Preprocessing

The raw data from the repository is converted to CSV. It also has a messy data column with mixed formats, and there are some rows with NaN values. The dates are converted from %m/%d/%Y, %d-%m-%Y, and %d/%m/%Y to %d/%m/%Y.

| Sampling Date |     |     |     |           
|---------------|-----|-----|-----|          
| 1/7/2014      |     |     |     |           
| 21-01-14      |     |     |     |       
| 2/4/2014      |     |     |     |           
| 2/6/2014      |     |     |     |           
| 2/11/2014     |     |     |     |               
| 02-13-14      |     |     |     |           

The data is first cleaned and preprocessed using Pandas to regularise the date formats and interpolate the missing values. 

| Sampling Date |     |     |     |
|---------------|-----|-----|-----|
| 07/01/2014    |     |     |     |
| 21/01/2014    |     |     |     |
| 04/02/2014    |     |     |     |
| 06/02/2014    |     |     |     |
| 11/02/2014    |     |     |     | 
| 13/02/2014    |     |     |     | 

Since the dataset is available yearwise, we also need to append all of the cleaned files to create our training set.

![img1](https://github.com/dnezan/air-quality-forecasting-azureAutoML/blob/master/img/data_no_interpolation.png?raw=true)
*Dataset before interpolation*

![img2](https://github.com/dnezan/air-quality-forecasting-azureAutoML/blob/master/img/data_interpolation.png?raw=true)
*Dataset after interpolation*

![img3](https://github.com/dnezan/air-quality-forecasting-azureAutoML/blob/master/img/thiruv_daily_data.png?raw=true)
*Daily NO2 data for Thiruvottiyur*


## Setting up Azure AutoML

Azure AutoML is a service by Azure that can create a number of parallel pipelines that execute various algorithms. It iterates through a selection of algorithms where each model is produced along with its respective training score. It has numerous advantages, the most prominent of which is time saving and computation cost, since it can automatically perform feature engineering, model evaluation, model deployement, and other tasks related to time series forecasting.

To set up Azure AutoML, we use the AutoML SDK v2, but you can also set up your experiments by using the GUI on the Azure Portal.

1. Create your ML_client connection using your Azure workspace credentials.
2. Import training and testing data in MLTable format.
3. Create forecasting experiment along with various training properties. We can specify holidays, what algorithms to block, and other training configurations like timeout and forecast window. Since we are only training on a specific number of algorithms, we need to block all of the others. We also need to enable deep learning in order to make use of the TCN Deep Neural Network Forecaster.
4. Begin the experiment. Once the models have been trained, you can compare model performance using RMSE (root mean square error). 
5. Optionally, you can deploy your model as an endpoint in order to use it in an application.

## Model Evaluation

We compare the performance between AutoARIMA, Prophet, and TCNForecaster.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
x = ['Prophet', 'autoARIMA', 'TCN Forecaster']
y = [prophet_mape, autoARIMA_mape, tcn_mape]
ax.bar(x, y, width=0.4)
ax.set_xlabel('Regressor models')
ax.set_ylabel('RMSE')
ax.set_ylim(0, 1)

plt.show()
```

![rmse](https://raw.githubusercontent.com/dnezan/air-quality-forecasting-azureAutoML/master/img/rmse.png?token=GHSAT0AAAAAAB7WFMCTAC7TBACZOPGW4STGZBNLU4A)

Let us predict the value of NO2 emissions for Kathivakkam, Municipal Kalyana Mandapam, Chennai using our new model.

![forecast](https://raw.githubusercontent.com/dnezan/air-quality-forecasting-azureAutoML/master/img/forecast_100days.png?token=GHSAT0AAAAAAB7WFMCTF7P6MIV6G5ZZZIJWZBNLVSA)
*Forecast window = 100 days*

## Inferences

To calculate the Air Quality Index (AQI) in India, you need to have the concentrations of all eight pollutants, including PM2.5, ozone (O3), carbon monoxide (CO), ammonia (NH3), and lead (Pb), in addition to SO2, NO2, and SPM/PM10. However, if you only have the concentrations of SO2, NO2, and SPM/PM10, you can still calculate the AQI using a simplified method.

The simplified method involves the following steps:

1. Calculate the sub-index for each pollutant based on its concentration using the following formulas:
	- For SPM/PM10:
	  SPM/PM10 sub-index = (SPM/PM10 concentration / 100) x 50
	- For NO2:
	  NO2 sub-index = (NO2 concentration / 80) x 50
	- For SO2:
	  SO2 sub-index = (SO2 concentration / 80) x 40
2. Identify the highest sub-index among the three pollutants.
3. Report the AQI based on the highest sub-index, using the following table:
   AQI = (Highest sub-index / 50) x 100
   Range of AQI values:

| AQI             | Rating                                                                |
| --------------- | ------------------------------------------------------------------ |
| 0-50    | ![Good](https://via.placeholder.com/10/00ff00?text=+) Good |
| 51-100  | ![Satisfactory](https://via.placeholder.com/10/7fff00?text=+) Satisfactory |
| 101-200 | ![Moderate](https://via.placeholder.com/10/ffff00?text=+) Moderate |
| 201-300 | ![Poor](https://via.placeholder.com/10/ffa500?text=+) Poor |
| 301-400 | ![Very Poor](https://via.placeholder.com/10/ff4500?text=+) Very Poor |
| 401-500 | ![Severe](https://via.placeholder.com/10/ff0000?text=+) Severe |

Note that this method is only an approximation and may not be as accurate as the official CPCB method, which takes into account the concentrations of all eight pollutants.

Using these labels, this problem statement can also be extended into a classification problem in order to predict future air quality index in Chennai.

## References

 - [AutoML Forecasting Bike Share](https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/automl-standalone-jobs/automl-forecasting-task-bike-share/auto-ml-forecasting-bike-share.ipynb)