# 🏦 Predictive Analysis of Client Demographics & Behavior in Banking Campaigns

An exploratory data analysis (EDA) project on a Portuguese bank's telemarketing campaign dataset, uncovering patterns in client demographics, financial behavior, and campaign interactions to understand what drives term deposit subscriptions.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Analysis Questions Covered](#analysis-questions-covered)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)

---

## 📌 Overview

This project performs a deep **Exploratory Data Analysis (EDA)** on a real-world banking marketing dataset. The goal is to analyze **45,216 client records** across 17 attributes — demographics, financial status, and campaign history — to identify patterns that predict whether a client will subscribe to a **term deposit**.

The analysis answers 18 structured business questions using statistical summaries and visualizations, building a complete picture of client behavior and campaign effectiveness.

---

## 📂 Dataset

- **Domain:** Portuguese bank telemarketing campaigns
- **Records:** 45,216 clients
- **Target Variable:** `y` — whether the client subscribed to a term deposit (`yes` / `no`)
- **Source:** UCI Bank Marketing Dataset

### Feature Categories

| Category | Features |
|---|---|
| Client Demographics | `age`, `job`, `marital`, `education` |
| Financial Status | `default`, `balance`, `housing`, `loan` |
| Campaign Contact | `contact`, `day`, `month`, `duration`, `campaign` |
| Previous Campaign | `pdays`, `previous`, `poutcome` |
| Target | `y` (term deposit subscription) |

---

## 🔍 Analysis Questions Covered

| # | Question |
|---|---|
| Q1 | What is the distribution of age among the clients? |
| Q2 | How does job type vary among the clients? |
| Q3 | What is the marital status distribution? |
| Q4 | What is the level of education among clients? |
| Q5 | What proportion of clients have credit in default? |
| Q6 | What is the distribution of average yearly balance? |
| Q7 | How many clients have housing loans? |
| Q8 | How many clients have personal loans? |
| Q9 | What communication types were used during the campaign? |
| Q10 | What is the distribution of last contact day of the month? |
| Q11 | How does last contact month vary among clients? |
| Q12 | What is the distribution of last contact duration? |
| Q13 | How many contacts were performed per client during the campaign? |
| Q14 | Distribution of days passed since last contact from a previous campaign? |
| Q15 | How many contacts were performed before the current campaign? |
| Q16 | What were the outcomes of previous marketing campaigns? |
| Q17 | Distribution of clients who subscribed vs. those who did not? |
| Q18 | Are there correlations between attributes and subscription likelihood? |

---

## 📊 Key Findings

### 👥 Client Demographics
- **Age** ranges from 18 to 95, with a mean of **40.9 years** (std: 10.6) — majority between 30–50
- **Top job types:** Blue-collar (9,732), Management (9,460), Technician (7,597)
- **Marital status:** Married (60.2%), Single (28.3%), Divorced (11.5%)
- **Education:** Secondary (51.3%), Tertiary (29.4%), Primary (15.2%)

### 💰 Financial Profile
- Only **1.80%** of clients have credit in default (815 out of 45,216)
- **Average yearly balance** — mean: €1,362, median: €448.5 — highly right-skewed with outliers up to €102,127
- **55.6%** of clients have a housing loan (25,130)
- **16.0%** have a personal loan (7,244)

### 📞 Campaign Behavior
- **Contact types:** Cellular (64.8%), Unknown (28.8%), Telephone (6.4%)
- **Most contacts happened in May** (13,766 clients — over 2× any other month)
- Average **2.76 contacts per client** during the campaign (max: 63)
- Most clients (75th percentile) had **not been contacted before** (`pdays = -1`)

### 🎯 Previous Campaign Outcomes
- 81.7% of previous outcomes are **unknown** (36,961) — first-time contacts
- Only 1,513 clients (3.3%) had a **successful** previous campaign outcome

### 🏆 Target Variable — Term Deposit Subscription
- **88.3%** did NOT subscribe (39,922)
- **11.7%** subscribed (5,294) — significant class imbalance

### 🔗 Top Correlations with Subscription (`y_yes`)
| Feature | Correlation |
|---|---|
| `duration` (call length) | **+0.394** |
| `poutcome_success` (prev. success) | **+0.307** |
| `month_mar` | +0.129 |
| `month_oct` | +0.128 |
| `housing_yes` | -0.139 |
| `contact_unknown` | -0.151 |
| `poutcome_unknown` | -0.167 |

> 💡 **Key insight:** Call duration and previous campaign success are the strongest predictors of subscription — longer calls and prior successes significantly increase conversion probability.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat)

| Library | Usage |
|---|---|
| `pandas` | Data loading, filtering, value counts, describe() |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting |
| `seaborn` | `histplot`, `countplot`, `heatmap` visualizations |
| `sklearn` | `get_dummies` for one-hot encoding (correlation analysis) |

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/pawanahirwa/Predictive-Analysis-of-Client-Demographics-and-Behavior-in-Banking-Campaigns.git
cd Predictive-Analysis-of-Client-Demographics-and-Behavior-in-Banking-Campaigns
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Download the dataset
Get the Bank Marketing dataset from [UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) or [Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset) and load it:
```python
import pandas as pd
data = pd.read_csv('bank.csv')   # or 'bank-full.csv'
```

### 4. Run the notebook
```bash
jupyter notebook banking_eda.ipynb
```

---

## 📁 Project Structure

```
Predictive-Analysis-Banking-Campaigns/
│
├── banking_eda.ipynb      # Main EDA notebook — 18 analysis questions
├── bank.csv               # Dataset (download separately from UCI/Kaggle)
└── README.md
```

---

## 👤 Author

**Pawan Singh Ahirwar**
- GitHub: [@pawanahirwa](https://github.com/pawanahirwa)
- Affiliation: IRCC, IIT Bombay
