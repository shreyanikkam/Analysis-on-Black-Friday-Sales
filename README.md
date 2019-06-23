# Analysis-on-Black-Friday-Sales
In this project, we determine the customer purchase behavior against different products. This analysis would help stores find targeted customers to increase their sales and purchases at a larger scale.

LINK TO DATASET:

https://drive.google.com/open?id=1IPcOuabwyZh_Tv0mtz-C6EVioEF1J3B8


Characteristics of Dataset:
1. Dataset of 550 000 observations about the black Friday in a retail store.
2. It has 550k rows x 12 columns which is available in the .csv format
3. It contains different kinds of variables either numerical or categorical. 
4. It contains missing values.

Problem Defintion:


1.During the Black Friday sales, finding the targeted customers would be a greatest advantage to increase the sales and purchases. 
2. By analysis, we can find the specific group of consumers at which a company aims its products and services which gives you target customers who are most likely to buy from you.
3. This will influence the marketing strategies and profit by concentrating on advertising their products to their targeted customer at larger scale.
4. How are transactions distributed over different age groups, occupations and cities.
5. How does marital status and years of living in the city affect number and amount of purchases?

Implementation:


We imported the seaborn, IPython and jinja2 package to run the code. The snapshots below shows the implementation of the data mining techniques used in our project:

1.	EDA
The analysis in the below graphical representation shows the amount of purchases made by men and women, category of occupations and age groups during the Black Friday Sales. 


RMSE around 3236.14 mean of the target value based on Product_ID.

Conclusion:


We found the targeted customers where the customers having occupation category (0,4,7); age group ranging 26-35, staying in City B, and for 1year, who are single make more purchases than the average. In this project we performed the exploratory data analysis and Random Forest Regression to predict purchase amount based on user id, product id and other features available in dataset. Our model had RMSE around 3236.14 mean of the target value based on Product_ID. The customers are clustered using K-Means algorithm with 3 clusters and a silhouette score of 0.09.





