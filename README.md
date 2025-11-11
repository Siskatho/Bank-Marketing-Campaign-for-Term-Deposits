# Bank Marketing Term Deposit Campaign Prediction

## 1. Project Overview

This project analyzes and predicts customer responses to term deposit telemarketing campaigns conducted by a commercial bank in Portugal. Term deposits represent a stable and long-term funding source for the bank, but traditional telemarketing campaigns are often **costly and inefficient** when customers are targeted broadly.

To improve campaign effectiveness, this project identifies customer characteristics that influence deposit decisions and builds a **machine learning model** that allows the bank to **prioritize high-potential customers**, reduce wasted calls, and improve conversion rates.

### Objectives
- Identify key features influencing customer deposit subscription.
- Handle class imbalance and optimize machine learning models.
- Improve marketing efficiency through targeted outreach.
- Support business decision-making with interpretable model insights.

---

## 2. Data Sources

- **Bank Marketing Campaigns Dataset (UCI ML Repository)**  
  https://archive.ics.uci.edu/ml/datasets/bank+marketing  
- Data consists of customer attributes, campaign interactions, historical contact outcomes, and macroeconomic indicators.

**Dataset Size:** 41,188 rows × 21 columns  
**Target Variable:** `y` — whether the customer subscribed to a term deposit (`yes` / `no`).

---

## 3. Technologies Used

| Category | Tools / Libraries |
|---------|-------------------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, Imbalanced-learn (SMOTENC) |
| Model Interpretation | Logistic Regression, Odds Ratio Analysis |
| Development Environment | Jupyter Notebook |

---

## 4. Project Structure
  
  ├── README.md
  
  ├── data
  
  │   ├── raw
  
  │   └── cleaned
  
  ├── notebooks
  
  │   └── eda_and_modeling.ipynb
  
  ├── reports
  
  │   ├── slides
  
  │   └── figures
  
  ├── models
  
  │   └── final_model.pkl
  
  └── src
  
      ├── preprocessing.py
      
      ├── modeling.py
      
      └── utils.py


---

## 5. Key Insights from Analysis

### Customer Behavior Insights
- Only **~11%** of customers subscribed → **high class imbalance**.
- **Students and retirees** show the highest subscription likelihood.
- Calls **via cellular** significantly outperform landline calls.
- More than **3 contact attempts** decreases success rate → overcalling reduces effectiveness.
- Customers with **successful past campaign outcomes** are strong positive prospects.

### Economic Context Insights
- Lower consumer confidence and lower employment variation are associated with higher deposit subscription rates — customers seek **financial security in uncertain economic conditions**.

---

## 6. Machine Learning Approach

### Preprocessing Steps
- Removed invalid or inconsistent values.
- Handled categorical features via:
  - One-Hot Encoding
  - Ordinal Encoding (for ordered education levels)
  - Binary Encoding (for high-cardinality categorical fields)
- Addressed class imbalance using **SMOTENC** to synthesize samples correctly for mixed data types.
- Removed multicollinearity and features causing data leakage (`duration`, highly correlated macro indicators).

### Model Training & Evaluation
Multiple models were compared using **Cross-Validation (F2-score)** due to higher cost of missing deposit-positive customers (false negatives):

| Model | Performance Summary |
|------|---------------------|
| Logistic Regression | ✅ Selected — best interpretability + stable performance |
| LDA | Slightly higher F2 but less interpretable |
| SVC | Comparable performance but harder to explain to stakeholders |
| Tree-Based Models | Accurate but lower interpretability |

**Final Model:** Logistic Regression (L1 Regularization)  
**Performance (F2-score):** ≈ 0.53  

### Key Predictive Features (based on Odds Ratios)
| Feature | Effect |
|--------|--------|
| Previous successful campaign | **Strong positive predictor** |
| Contact method = cellular | Higher conversion probability |
| High number of campaign calls | **Negative impact** |
| Month of contact | Seasonal trend impacts success |
| Job = student or retired | Higher likelihood to subscribe |

---

## 7. Business Recommendations

| Recommendation | Impact |
|---------------|--------|
| Prioritize customers with previous successful campaign history | Highest ROI targeting |
| Use **cellular** as the primary contact channel | Higher conversion efficiency |
| Limit call attempts to **1–3 per customer** | Avoids wasted operational cost |
| Focus campaigns during **March, June, July** | Seasonal uplift in interest |
| Target segments: **students and retirees** first | Highest responsiveness |

---

## 8. Deployed Model

Model is deployed through streamlit

[https://bmctermdeposit-aey9jpu8i3rvdviogh8fer.streamlit.app](https://bmc-deployed.streamlit.app)

---

## 9. Tableau

https://public.tableau.com/app/profile/fransiska.sri.mayawi/viz/TableauFinalProject_17594030848810/Dashboard1?publish=yes

![tableau](tableau.jpeg)

---

## 10. Contact

For further discussion or collaboration:

**Name:** *Muhammad Nafi Adziq & Fransiska Sri Mayawi*  
**Email:** *muhammadnafiqadziq@gmail.com & siskatho17@gmail.com*  
**LinkedIn:** *[Muhammad Nafi Adziq](https://www.linkedin.com/in/muhammad-nafi-adziq-19ab0936a/) & [Fransiska Sri Mayawi](https://www.linkedin.com/in/fransiskasrimayawi/)*
