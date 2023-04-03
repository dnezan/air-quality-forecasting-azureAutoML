![banner](https://raw.githubusercontent.com/dnezan/air-quality-forecasting-azureAutoML/master/img/repo_banner.png)
# Air Quality Forecasting using Azure AutoML in Python

This project focuses on using Azure AutoML and Python SDK to build a time series forecasting model for air quality in my home city, Chennai. By leveraging AutoML's automated machine learning capabilities and the flexibility of Python SDK, the model aims to predict future air quality with high accuracy, enabling government agencies and individuals to make data-driven decisions to combat air pollution.

The research dataset was sourced from [Open Government Data (OGD) Platform India - Tamil Nadu](https://tn.data.gov.in/catalog/historical-daily-ambient-air-quality-data) and underwent cleaning and preparation after being downloaded.

## Environment Variables

To run this project in Azure AutoML, you will need to add the following environment variables to your config.json file

`subscription_id`

`resource_group`

`workspace-name`


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

