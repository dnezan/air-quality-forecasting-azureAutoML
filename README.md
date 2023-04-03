![banner](https://raw.githubusercontent.com/dnezan/air-quality-forecasting-azureAutoML/master/img/repo_banner.png)
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

| AutoARIMA           | Decision Tree               | ForecastTCN          | Naive           |
| Prophet             | Arimax                      | Gradient Boosting    | SeasonalAverage |
| Elastic Net         | LARS Lasso                  | ExponentialSmoothing |                 |
| Light GBM           | Extremely Randomized Trees* | SeasonalNaive        |                 |
| K Nearest Neighbors | Random Forest               | Average              |                 |

## Environment Variables

To run this project in Azure AutoML, you will need to add the following environment variables to your config.json file

`subscription_id`

`resource_group`

`workspace-name`

## Dataset

Throughtout the years, the number of monitoring stations in Tamil Nadu has increased, as well as their capabilities. The very early years of 1987 had very few agencies that were only able to detect a few types of pollutants. In 2015, we now have data on SO2, NO2, RSPM/PM10, and SPM levels across 11 different monitoring agencies.


|                                                  | 2010  | 2011  | 2012  | 2013  | 2014  | 2015  |
|--------------------------------------------------|-------|-------|-------|-------|-------|-------|
| Kathivakkam, Municipal Kalyana Mandapam, Chennai | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |
| Govt. High School, Manali, Chennai.              | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |
| Thiruvottiyur, Chennai                           | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |
| Madras Medical College, Chennai                  | **Y** | **Y** | **Y** | N     | **Y** | **Y** |
| NEERI, CSIR Campus Chennai                       | **Y** | **Y** | **Y** | N     | **Y** | **Y** |
| Thiruvottiyur Municipal Office, Chennai          | **Y** | **Y** | **Y** | N     | **Y** | **Y** |
| Adyar, Chennai                                   | N     | N     | N     | **Y** | **Y** | **Y** |
| Anna Nagar, Chennai                              | N     | N     | N     | **Y** | **Y** | **Y** |
| Thiyagaraya Nagar, Chennai                       | N     | N     | N     | **Y** | **Y** | **Y** |
| Kilpauk, Chennai                                 | N     | N     | N     | **Y** | **Y** | **Y** |
| Vallalar Nagar, Chennai                          | N     | N     | N     | N     | N     | **Y** |

We will be forecasting air pollution in three different areas due to the availability of the data (Kathivakkam, Manali, and Thiruvottiyur) in the years 2010-2015. Coincidentally, all three areas fall under Industrial locations and will therefore see pollutant levels that are higher than that of the surrounding residential areas.


## Data Preprocessing

The raw data from the repository is converted to CSV. It also has a messy data column with mixed formats, and there are some rows with NaN values.

| Sampling Date |     |     |     |     |
|---------------|-----|-----|-----|-----|
| 1/7/2014      |     |     |     |     |
| 21-01-14      |     |     |     |     |
| 2/4/2014      |     |     |     |     |
| 2/6/2014      |     |     |     |     |
| 2/11/2014     |     |     |     |     |
| 02-13-14      |     |     |     |     |

The data is first cleaned and preprocessed using Pandas to regularise the date formats and interpolate the missing values. Since the dataset is available yearwise, we also need to append all of the cleaned files to create our training set.


## References

 - [AutoML Forecasting Bike Share](https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/automl-standalone-jobs/automl-forecasting-task-bike-share/auto-ml-forecasting-bike-share.ipynb)