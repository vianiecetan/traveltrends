# Travel Trends in Singapore
This github repository consists of our machine learning project for module code INF2008.

## Problem Statement
This study aims to predict popular travel destinations during holidays and major events by analyzing air passenger departures and arrivals in Singapore. By examining historical flight data, passenger trends, and external factors such as public holidays and major global or regional events, this research seeks to identify patterns in travel behavior. Understanding these trends can provide valuable insights for airlines, travel agencies, and policymakers to optimize flight schedules, manage demand, and enhance tourism strategies. Additionally, this study will explore the influence of factors such as destination appeal to refine predictions and improve forecasting models.

## Dataset
Our datasets that we decided to use are from the department of statistics singapore (singstat.gov.sg).

Departure: https://tablebuilder.singstat.gov.sg/table/TS/M650641

Arrivals: https://tablebuilder.singstat.gov.sg/table/TS/M650631

Inflation: 

## EDA

In this section, we will perform Exploratory Analysis Data to gain insights on the relationship between features and the data itself to justify if the features are useful.

### Correlation Heat Map 

<img src="images/corelation.png" alt="Alt text" width="500"/>

This graph is to present linear relationships between features. However after anlysising it, it shows does not show a strong relationship to one another. 

### Scatter Plot Holidays vs Total Traffics 

<img src="images/SP_holidays-vs-traffic.png" alt="Alt text" width="500"/>

This analysis is to show the relationship and data dsitribution of total traffic across the years against number of public holidays. Although it does not show strong relationship or pattern, we can tell that more traffic resides during lesser holidays.

### Scatter Plot Inflation vs Traffic 

<img src="images/SP_Inflation-vs-Traffic.png" alt="Alt text" width="500"/>

This analysis is to show the relationship and data dsitribution of total traffic against inflation rating. This shows a strong relationship where more flights occurs when inflation were low.

### Line Plot Arrivals last 10 years

<img src="images/LP_Arr-last-10.png" alt="Alt text" width="500"/>

This analysis is to show the relationship and data dsitribution of total traffic against inflation rating. This shows a strong relationship where more flights occurs when inflation were low.


## Algorithms

