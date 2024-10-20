# [Python] Customer Segmentation with RFM Analysis
## **I. INTRODUCTION**
### 1. BUSINESS QUESTIONS
SuperStore, a global retail company, plans to launch marketing campaigns for the holiday season to reward loyal customers and target potential ones. Due to the large volume of data, manual customer segmentation is no longer feasible. The Marketing Director proposed using the RFM model, and the Data Analytics team has been asked to automate this process using Python.

This project utilizes Python to segment customers, providing insights into the characteristics of each segment and offering actionable recommendations for the Marketing team to tailor their campaigns effectively.

### 2. DATASET
Dataset used (as attachment) is a transnational dataset which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

|Field Name| Detail|
|---|---|
|InvoiceNo |Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'C', it indicates a cancellation.|
|StockCode |Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.|
|Description| Product (item) name. Nominal.|
|Quantity| The quantities of each product (item) per transaction. Numeric.|
|InvoiceDate| Invoice Date and time. Numeric, the day and time when each transaction was generated.|
|UnitPrice| Unit price. Numeric, Product price per unit in sterling.|
|CustomerID| Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.|
|Country| Country name. Nominal, the name of the country where each customer resides.|

### 3. ABOUT RFM ANALYSIS
***What is RFM Analysis?***
- RFM is a marketing analysis technique that stands for Recency, Frequency, and Monetary Value.
    - **Recency**: measures how recently a customer has made a purchase.
    - **Frequency**: measures how often a customer has made purchases.
    - **Monetary Value**: measures the total amount of money a customer has spent on purchases.

***Why RFM Analysis?***
- RFM is a part of Marketing Analysis and is used to evaluate customer value, helping businesses analyze and segment their customer base. This allows for tailored marketing campaigns or special customer care initiatives for each group.

***How does RFM Analysis work?***
- In RFM analysis, customers are scored based on three factors (Recency - how recently, Frequency - how often, Monetary - how much), then labeled (segmented) based on the combination of RFM scores

## **II. DATA PREPARATION (EDA)**
In this data preparation stage, we will check for missing values, duplicates, and incorrect data types/data values to make sure the dataset is clean and ready for further analysis.
### 1. CHECKING
```Python
# Checking for Missing Values
print(ecommerce.isna().sum())

# Checking for Duplications
print(ecommerce.shape)
print(ecommerce.nunique())

# Checking data type
ecommerce.info()

# Checking data value
ecommerce.describe()
```
### 2. HANDLING
- Missing Values:
  - 1454 rows in Description -> No base to fill in -> No action
  - 135080 rows in CustomerID -> No base to fill in -> No action
- Duplicates:
  - All columns have duplicates -> Table have no PK (Because 1 invoice having many purchased items, each line represents 1 item) -> No action
- Data Type:
  - InvoiceDate object -> datetime
- Data Values:
  - Quantity < 0 -> Cancelled invoices -> Remove
  - UnitPrice < 0 -> Assumption:Error -> Remove
```Python
# Changing data type of InvoiceDate
ecommerce['InvoiceDate']= pd.to_datetime(ecommerce['InvoiceDate'])

# Remove non-delivered invoices
filter = ecommerce[(ecommerce['Quantity'] <= 0) | (ecommerce['UnitPrice'] <= 0)].index
ecommerce.drop(filter, inplace=True)
```
## **III. RFM ANALYSIS & SEGMENTATION**
### 1. RFM CALCULATION
Calculating R,F,M respectively based on its definition:
  - **Recency**: number of days since last purchase of each customer.
  - **Frequency**: total number of transactions of each customer.
  - **Monetary Value**: total spending of each customer.
```Python
# Calculating R
  # Find last purchase date of each customer
  calculation = ecommerce.groupby("CustomerID").agg({"InvoiceDate":"max"})
  # Calculate number of days from last purchase
  from datetime import datetime
  calculation["CurrentDate"] = datetime(2011, 12, 31)
  calculation["Recency"] = calculation["InvoiceDate"] - calculation["CurrentDate"]

# Calculating F
  calculation["Frequency"] = ecommerce.groupby("CustomerID")["InvoiceNo"].nunique()

# Calculating M
  # Calculate line total of each item in each invoice
  ecommerce["LineTotal"] = ecommerce["Quantity"]*ecommerce["UnitPrice"]
  # Calculate total purchase value per customer
  calculation["Monetary"] = ecommerce.groupby("CustomerID")["LineTotal"].sum()

calculation.head()
```
|CustomerID |InvoiceDate	|CurrentDate	|Recency	|Frequency	|Monetary|
|---|---|---|---|---|---|
|12346.0	|2011-01-18 10:01:00	|2011-12-31	|-347 days +10:01:00	|1	|77183.60|
|12347.0	|2011-12-07 15:52:00	|2011-12-31	|-24 days +15:52:00	|7	|4310.00|
|12348.0	|2011-09-25 13:13:00	|2011-12-31	|-97 days +13:13:00	|4	|1797.24|
|12349.0	|2011-11-21 09:51:00	|2011-12-31	|-40 days +09:51:00	|1	|1757.55|
|12350.0	|2011-02-02 16:01:00	|2011-12-31	|-332 days +16:01:00	|1	|334.40|

### 2. RANKING
Quintiles were used to assign scores to each RFM component of each customer. Then all separated R,F,M score were concated into 1 RFM score, which is foundation for segmentation stage and further analysis.
```Python
# Rank order of data to define cut point
orderFrequency = calculation["Frequency"].rank(method='first')

# Scoring R-F-M
calculation["F_score"] = pd.qcut(orderFrequency, 5, labels=["1", "2", "3", "4", "5"])
calculation[["R_score", "M_score"]] = calculation[["Recency", "Monetary"]].apply(lambda x: pd.qcut(x, 5, labels=["1", "2", "3", "4", "5"]))

# Concat R-F-M score
calculation["RFM_score"] = calculation.apply(lambda x:'%s%s%s' % (x["R_score"],x["F_score"],x["M_score"]),axis=1)
calculation["RFM_score"] = calculation["RFM_score"].astype(int)

calculation = calculation.reset_index()
calculation.head()
```
|CustomerID |InvoiceDate	|CurrentDate	|Recency	|Frequency	|Monetary|F_score	|R_score	|M_score	|RFM_score|
|---|---|---|---|---|---|---|---|---|---|
|12346.0	|2011-01-18 10:01:00	|2011-12-31	|-347 days +10:01:00	|1	|77183.60|1|	1	|5	|115|
|12347.0	|2011-12-07 15:52:00	|2011-12-31	|-24 days +15:52:00	|7	|4310.00|5|	5	|5	|555|
|12348.0	|2011-09-25 13:13:00	|2011-12-31	|-97 days +13:13:00	|4	|1797.24|4|	2|	4|	244|
|12349.0	|2011-11-21 09:51:00	|2011-12-31	|-40 days +09:51:00	|1	|1757.55|1|	4	|4|	414|
|12350.0	|2011-02-02 16:01:00	|2011-12-31	|-332 days +16:01:00	|1	|334.40|1|	1|	2|	112|

### 3. SEGMENTATION
Segment customers of SuperStore based on 11 pre-defined groups.
```Python
# Convert comma-separated string to a list of RFM scores
segmentation["RFM Score"] = segmentation["RFM Score"].str.split(",")

# Transform each element of a list-like to a row
segmentation = segmentation.explode("RFM Score").reset_index(drop=True)
segmentation["RFM Score"] = segmentation["RFM Score"].astype(int)

# Merge segmentation with calculation df to show Segment name
rfm = calculation.merge(segmentation, how="left", left_on="RFM_score", right_on="RFM Score")
rfm.head()
```
|CustomerID |InvoiceDate	|CurrentDate	|Recency	|Frequency	|Monetary|F_score	|R_score	|M_score	|RFM_score| Segment|
|---|---|---|---|---|---|---|---|---|---|---|
|12346.0	|2011-01-18 10:01:00	|2011-12-31	|-347 days +10:01:00	|1	|77183.60|1|	1	|5	|115| Cannot Lose Them|
|12347.0	|2011-12-07 15:52:00	|2011-12-31	|-24 days +15:52:00	|7	|4310.00|5|	5	|5	|555| Champions|
|12348.0	|2011-09-25 13:13:00	|2011-12-31	|-97 days +13:13:00	|4	|1797.24|4|	2|	4|	244| At Risk|
|12349.0	|2011-11-21 09:51:00	|2011-12-31	|-40 days +09:51:00	|1	|1757.55|1|	4	|4|	414| Promising|
|12350.0	|2011-02-02 16:01:00	|2011-12-31	|-332 days +16:01:00	|1	|334.40|1|	1|	2|	112| Lost customers|

## **IV. VISUALIZATION**
### 1. DISTRIBUTION OF R,F,M
```Python
colnames = ["Recency","Frequency","Monetary"]

for col in colnames:
  fig,ax = plt.subplots(figsize=(12,3))
  sns.distplot(rfm[col])
  ax.set_title("Distribution of %s" % col)
plt.show()
```
<img width="600" alt="Dist of RFM" src="https://github.com/user-attachments/assets/712d195e-3f7a-4602-9635-b87f5ff06e97">

### 2. SEGMENT BY CUSTOMER COUNT
```Python
# Count number of customers per Segment
grp = rfm.groupby("Segment").agg({"CustomerID":"count"})
grp = grp.reset_index()
grp["Percent"] = grp["CustomerID"]/ (grp["CustomerID"].sum())

# Import libraries
!pip install squarify
import squarify

# Define colors
colors = ["#9e0142","#d53e4f","#f46d43","#fdae61","#fee08b","#ffffbf","#e6f598","#abdda4","#66c2a5","#3288bd","#5e4fa2"]

# Draw treemap
fig,ax = plt.subplots(1, figsize=(10,5))
squarify.plot(sizes=grp["CustomerID"],
              label=grp["Segment"],
              value=[f'{x*100:.2f}%' for x in grp["Percent"]],
              alpha=.8,
              color=colors,
              bar_kwargs=dict(linewidth=1.5, edgecolor="white")
              )
plt.title("Customer Size by Segment", fontsize=13)
plt.axis("off")
plt.show()
```
<img width="600" alt="Seg by cust count" src="https://github.com/user-attachments/assets/a25d15d8-f6a3-4f5d-b64e-0ff8d769beff">

### 3. SEGMENT BY TOTAL SALES
```Python
# Calculate Total Sales per Segment
grpM = rfm.groupby("Segment").agg({"Monetary":"sum"})
grpM = grpM.reset_index()
grpM["Percent"] = grpM["Monetary"]/ (grpM["Monetary"].sum())

fig,ax = plt.subplots(1, figsize=(10,5))
squarify.plot(sizes=grpM["Monetary"],
              label=grpM["Segment"],
              value=[f'{x*100:.2f}%' for x in grpM["Percent"]],
              alpha=.8,
              color=colors,
              bar_kwargs=dict(linewidth=1.5, edgecolor="white")
              )
plt.title("Total Sales by Segment", fontsize=13)
plt.axis("off")
plt.show()
```
<img width="600" alt="Seg by sales" src="https://github.com/user-attachments/assets/f203b918-b601-44a4-8996-231df37cc627">

## **V. INSIGHTS & RECOMMENDATIONS**
### 1. SEGMENTS & CHARACTERISTICS
Theoretically, there are 11 customer groups, each with characteristics and marketing strategies as below:
|Group	|Characteristics	|Suggested Solution|
|---|---|---|
|Champions|	New customers who make frequent, high-value purchases. Loyal and willing to spend generously, likely to purchase again soon.	|Retain at all costs + upsell to increase revenue.|
|Loyal|	Customers with moderate to high frequency but lower purchase value.	|Focus on increasing purchase value to convert them into Champions.|
|Potential Loyalist|	Recent customers with moderate purchase frequency and value, who have made multiple purchases.| Same as Loyal.|	
|New Customers|	Recently acquired customers with low purchase value and frequency.|	Ensure satisfaction in early transactions to encourage repeat, higher-value purchases.|
|Promising|	Recent customers with high purchase value but low frequency.	|Same as New Customers.|
|Need Attention|	Customers with moderate frequency and value who have not returned recently.	|Investigate reasons for disengagement and offer incentives to return.|
|At Risk|	Customers who have not returned in a while but used to buy frequently at a moderate value.|Same as Need Attention.|
|Cannot Lose Them|	High-value customers who haven't returned in a long time.	|Same as Need Attention.|
|About to Sleep|	Customers who haven’t bought in a while and used to purchase with low frequency and value.	|Encourage them to return once again with special offers.|
|Hibernating Customers|	Customers with low frequency and value, who haven’t returned in a long time.	|Same as About to Sleep|
|Lost Customers|	Customers who haven’t returned in a very long time with low purchase value and frequency.	|Same as About to Sleep|

### 2. SUPERSTORE'S SITUATION 
However, we need to base on SuperStore's customer segments status to have a tailored marketing strategy. SuperStore's situation as below:
- Currently,
  - ~20% of the company’s customers belong to the Champions group (the most loyal and highest-spending customers).
  - Another ~20% fall into the Loyal and Potential Loyalist groups.
  - These 3 groups collectively generate the majority of the company’s revenue (~76%).
- However,
  - 1/3 of customers are at risk of leaving (Hibernating Customers, Lost Customers, About to Sleep).
  - Around 18% of customers, classified as Need Attention, At Risk, and Cannot Lose Them, are also at risk if no action is taken.
- Meanwhile, potential new customers account for only ~10% in terms of numbers, contributing just 2% of total revenue.

### 3. PROPOSED MARKETING STRATEGY
Given the situation of SuperStore, the marketing team should focus on 3 main strategies:
|Group|	Strategy|
|:---:|---|
|Champions, Loyal, Potential Loyalist<br>`40% of customers` & `76% of revenue`|Maintain and upsell to generate even higher revenue.|
|Hibernate, Lost, About to Sleep<br>`34% of customers`|Launch a special promotion to encourage these customers to make 1 purchase again as soon as possible.|
|Need Attention, At Risk, Cannot Lose Them<br>`18% of customers` & `16% of revenue`|Investigate the reasons for disengagement and create tailored solutions to win them back (instead of letting them completely churn).|

### 4. PROPOSED MARKETING ACTIONS
Detailed Actions are suggested as below: 

***4.1. Champions, Loyal, Potential Loyalist:***

_a. Actions for Champions:_
- **Retention:**
  - Enhance customer service with loyalty programs, including special promotions and offers to show appreciation and accumulate points.
  - Focus on efficient customer support, ensuring prompt handling of inquiries, exchanges, and warranties.

- **Increase Revenue:**
  - Promote higher-value "you-may-also-like" products or product combos based on the following criteria:<br>(1) Items with a value greater than the average of those previously purchased by the Champions group.<br>(2) Products that rank among the top purchases frequently made by Champions.

  - Top 10 items met (1) & (2) criteria:
    ```Python
    # Find Champion customers
    champions = rfm[rfm["Segment"] == "Champions"]
    champions = champions[["CustomerID"]]
    
    # Find items purchased by Champions + their unit price
    item = champions.merge(ecommerce, on="CustomerID", how="left")
    item_price = item[["Description","UnitPrice"]]
    item_price = item_price.groupby("Description").agg({"UnitPrice":"mean"})
    item_price = item_price.reset_index()
    
    # Find avg price of items purchased by Champions
    mean_price = item_price["UnitPrice"].mean()
    
    # List of items purchased by Champions, having price > avg price of all items purchased by Champions
    itemlist1 = item_price[item_price["UnitPrice"]>= mean_price]

    # Find how many times an item is purchased by Champions customer
    itemlist2 = item[["Description","InvoiceNo"]]
    itemlist2 = itemlist2.groupby("Description").agg({"InvoiceNo":"count"}).sort_values("InvoiceNo", ascending=False)
    itemlist2 = itemlist2.reset_index()

    # List of items having price > avg price of items purchased by Champions & most requenctly purchased by Champions
    finlist = itemlist2.merge(itemlist1, on="Description", how="inner").sort_values(["InvoiceNo","UnitPrice"], ascending=[False,False])
    
    finlist.head(10)
    ```
    |   |Description|	InvoiceNo|	UnitPrice|
    |---|---|---|---|
    |0|	REGENCY CAKESTAND 3 TIER	|906|	12.429801|
    |1|	PARTY BUNTING	|752|	4.864654|
    |2|	POSTAGE	|656|	20.756052|
    |3|	SET OF 3 CAKE TINS PANTRY DESIGN|	615	|4.975024|
    |4|	SPOTTY BUNTING	|580|	4.904810|
    |5|	ALARM CLOCK BAKELIKE RED	|554|	3.779061|
    |6|	ALARM CLOCK BAKELIKE GREEN	|531|	3.813766|
    |7|	RETROSPOT TEA SET CERAMIC 11 PC	|454|	5.051938|
    |8|	HOT WATER BOTTLE KEEP CALM	|444|	4.847973|
    |9|	JAM MAKING SET WITH JARS	|439	|4.191253|
      
_b. Actions for Loyal & Potential Loyalist:_
- Offer promotions and incentives tied to specific spending thresholds for this group.
- Example:
  - **Tier 1**: Offer a free gift for transactions with a value > Average Order Value of the Loyal and Potential Loyalist group.  
    - Average order value of this group: `$377`
      
     ```Python
     # Find customers in Loyal & Potential Loyalist segment
     loyal = rfm[(rfm["Segment"] == "Loyal") | (rfm["Segment"] == "Potential Loyalist")]
     loyal = loyal[["CustomerID"]]
      
     # Filter invoices of Loyal & Potential Loyalist customers
     loyal_iv = loyal.merge(ecommerce, on="CustomerID", how="left")
      
     # Find the invoice value
     loyal_value = loyal_iv.groupby("InvoiceNo").sum("LineTotal")
      
     # Find the average invoice value
     loyal_avg = loyal_value['LineTotal'].mean()
     print(loyal_avg)
     ```
     ```
     377.22777710843366
     ```
      
  - **Tier 2**: Provide a higher-value free gift for transactions > Company's Average Order Value.  
    - Company's Average Order Value: `$534`
    
    ```Python
    # Find the value of all delivered invoices
    totalvalue = ecommerce.groupby("InvoiceNo").sum("LineTotal")
    
    # Find the average value of all delivered invoices
    avgvalue = totalvalue['LineTotal'].mean()
    print(avgvalue)
    ```
    ```
    534.403033266533
    ```

***4.2. Hibernate Customer, Lost Customer, About to sleep***
- The company needs to incentivize this group of customers to make another purchase as soon as possible through short-term retargeting campaigns, such as vouchers, discounts, or exclusive special offers.

***4.3. Need Attention, At risk, Cannot lose them***
- Instead of defaulting to broad discount and promotional campaigns to attract this group back, which may waste the company's budget, it is crucial to conduct surveys to identify the root causes of why these customers are not returning. Addressing these issues is key, as the reasons for their disengagement may stem from product quality, customer service quality, etc., rather than just pricing.
