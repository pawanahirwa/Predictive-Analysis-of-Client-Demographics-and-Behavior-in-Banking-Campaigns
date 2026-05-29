# ============================================================
# 🏦 Predictive Analysis of Client Demographics and Behavior
#    in Banking Campaigns
# ============================================================
# Dataset: Portuguese Bank Marketing Campaign (45,216 records)
# Goal: EDA to understand what drives term deposit subscriptions
# Download dataset: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Load Dataset ──────────────────────────────────────────────
data = pd.read_csv('bank.csv')

print("Dataset shape:", data.shape)
print("\nColumn names:", list(data.columns))
print("\nFirst 5 rows:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nMissing values:")
print(data.isnull().sum())


# ─────────────────────────────────────────────────────────────
# Q.1  What is the distribution of age among the clients?
# ─────────────────────────────────────────────────────────────

print(data['age'].describe())
print(data['age'].unique())

plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=30, kde=True)
plt.title('Distribution of Age among Clients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.2  How does the job type vary among the clients?
# ─────────────────────────────────────────────────────────────

print(data['job'].value_counts())
print(data['job'].unique())

plt.figure(figsize=(12, 8))
sns.countplot(y=data['job'], order=data['job'].value_counts().index,
              hue=data['job'], palette='viridis', legend=False)
plt.title('Distribution of Job Types among Clients')
plt.xlabel('Count')
plt.ylabel('Job Type')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.3  What is the marital status distribution of the clients?
# ─────────────────────────────────────────────────────────────

marital_status_counts = data['marital'].value_counts()
print(marital_status_counts)
print(data['marital'].unique())

plt.figure(figsize=(10, 6))
sns.countplot(x=data['marital'], order=data['marital'].value_counts().index,
              hue=data['marital'], palette='viridis', legend=False)
plt.title('Distribution of Marital Status among Clients')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.4  What is the level of education among the clients?
# ─────────────────────────────────────────────────────────────

education_counts = data['education'].value_counts()
print(education_counts)
print(data['education'].unique())

plt.figure(figsize=(10, 6))
sns.countplot(x=data['education'], order=data['education'].value_counts().index,
              hue=data['education'], palette='viridis', legend=False)
plt.title('Distribution of Education Levels among Clients')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.5  What proportion of clients have credit in default?
# ─────────────────────────────────────────────────────────────

default_counts = data['default'].value_counts()
print(default_counts)
print(data['default'].unique())

total_clients = len(data)
clients_with_default = default_counts['yes']
proportion_with_default = clients_with_default / total_clients
print(f"Proportion of clients with credit in default: {proportion_with_default:.2%}")

plt.figure(figsize=(7, 7))
plt.pie(default_counts, labels=default_counts.index, autopct='%1.1f%%',
        colors=['#3B528B', '#21908C'])
plt.title('Proportion of Clients with Credit in Default')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.6  What is the distribution of average yearly balance?
# ─────────────────────────────────────────────────────────────

balance_summary = data['balance'].describe()
print(balance_summary)
print(data['balance'].unique())

plt.figure(figsize=(10, 6))
sns.histplot(data['balance'], bins=30, kde=True)
plt.title('Distribution of Average Yearly Balance among Clients')
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.7  How many clients have housing loans?
# ─────────────────────────────────────────────────────────────

housing_loan_counts = data['housing'].value_counts()
print(housing_loan_counts)

num_housing_loans = housing_loan_counts['yes']
print(f"Number of clients with housing loans: {num_housing_loans}")

plt.figure(figsize=(8, 5))
sns.countplot(x=data['housing'], hue=data['housing'], palette='viridis', legend=False)
plt.title('Housing Loan Distribution among Clients')
plt.xlabel('Has Housing Loan')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.8  How many clients have personal loans?
# ─────────────────────────────────────────────────────────────

personal_loan_counts = data['loan'].value_counts()
print(personal_loan_counts)

num_personal_loans = personal_loan_counts['yes']
print(f"Number of clients with personal loans: {num_personal_loans}")

plt.figure(figsize=(8, 5))
sns.countplot(x=data['loan'], hue=data['loan'], palette='viridis', legend=False)
plt.title('Personal Loan Distribution among Clients')
plt.xlabel('Has Personal Loan')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.9  What communication types were used during the campaign?
# ─────────────────────────────────────────────────────────────

communication_types = data['contact'].unique()
print(communication_types)

communication_type_counts = data['contact'].value_counts()
print(communication_type_counts)

plt.figure(figsize=(8, 5))
sns.countplot(x=data['contact'], order=communication_type_counts.index,
              hue=data['contact'], palette='viridis', legend=False)
plt.title('Communication Types Used for Contacting Clients')
plt.xlabel('Contact Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.10  What is the distribution of last contact day of month?
# ─────────────────────────────────────────────────────────────

plt.figure(figsize=(14, 6))
day_order = sorted(data['day'].unique())
sns.countplot(x=data['day'], order=day_order,
              hue=data['day'], palette='viridis', legend=False)
plt.title('Distribution of Last Contact Day of the Month')
plt.xlabel('Day of Month')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.11  How does the last contact month vary among clients?
# ─────────────────────────────────────────────────────────────

contact_months = data['month'].unique()
print(contact_months)

contact_month_counts = data['month'].value_counts()
print(contact_month_counts)

plt.figure(figsize=(12, 6))
sns.countplot(x=data['month'], order=contact_month_counts.index,
              hue=data['month'], palette='viridis', legend=False)
plt.title('Distribution of Last Contact Month')
plt.xlabel('Month')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.12  What is the distribution of the last contact duration?
# ─────────────────────────────────────────────────────────────

print(data['duration'].describe())

plt.figure(figsize=(10, 6))
sns.histplot(data['duration'], bins=30, kde=True)
plt.title('Distribution of Duration of Last Contact')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.13  How many contacts were performed during the campaign?
# ─────────────────────────────────────────────────────────────

campaign_contacts_summary = data['campaign'].describe()
print(campaign_contacts_summary)

plt.figure(figsize=(12, 6))
campaign_data = data[data['campaign'] <= 20]
sns.countplot(x=campaign_data['campaign'], hue=campaign_data['campaign'],
              palette='viridis', legend=False)
plt.title('Number of Contacts Performed During the Campaign for Each Client')
plt.xlabel('Number of Contacts')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.14  Distribution of days passed since last previous contact
# ─────────────────────────────────────────────────────────────

pdays_summary = data['pdays'].describe()
print(pdays_summary)

plt.figure(figsize=(10, 6))
sns.histplot(data['pdays'], bins=30, kde=True)
plt.title('Distribution of Days Passed Since Last Contact from Previous Campaign')
plt.xlabel('Number of Days (-1 = not previously contacted)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

pdays_contacted = data[data['pdays'] != -1]['pdays']
print(f"Clients previously contacted: {len(pdays_contacted)} ({len(pdays_contacted)/len(data)*100:.1f}%)")
print(f"Clients NOT previously contacted: {(data['pdays'] == -1).sum()}")


# ─────────────────────────────────────────────────────────────
# Q.15  How many contacts before the current campaign?
# ─────────────────────────────────────────────────────────────

previous_contacts_summary = data['previous'].describe()
print(previous_contacts_summary)

plt.figure(figsize=(12, 6))
prev_data = data[data['previous'] <= 15]
sns.countplot(x=prev_data['previous'], hue=prev_data['previous'],
              palette='viridis', legend=False)
plt.title('Number of Contacts Performed Before the Current Campaign for Each Client')
plt.xlabel('Number of Previous Contacts')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.16  What were the outcomes of the previous campaigns?
# ─────────────────────────────────────────────────────────────

previous_outcomes = data['poutcome'].unique()
print(previous_outcomes)

previous_outcome_counts = data['poutcome'].value_counts()
print(previous_outcome_counts)

plt.figure(figsize=(10, 6))
sns.countplot(x=data['poutcome'], order=previous_outcome_counts.index,
              hue=data['poutcome'], palette='viridis', legend=False)
plt.title('Outcomes of Previous Marketing Campaigns')
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.17  Subscribed to term deposit vs. those who did not?
# ─────────────────────────────────────────────────────────────

subscription_counts = data['y'].value_counts()
print(subscription_counts)
print(f"Subscription rate: {subscription_counts['yes']/len(data)*100:.1f}%")

plt.figure(figsize=(8, 6))
sns.countplot(x=data['y'], hue=data['y'], palette='viridis', legend=False)
plt.title('Distribution of Clients Who Subscribed to a Term Deposit vs. Those Who Did Not')
plt.xlabel('Subscribed to Term Deposit')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 7))
plt.pie(subscription_counts, labels=['No', 'Yes'], autopct='%1.1f%%',
        colors=['#3B528B', '#5DC863'])
plt.title('Term Deposit Subscription Rate')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# Q.18  Correlations between attributes and subscription?
# ─────────────────────────────────────────────────────────────

# Encode categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Calculate full correlation matrix
correlation_matrix = data_encoded.corr()

# Extract correlations with target variable
target_correlation = correlation_matrix['y_yes'].sort_values(ascending=False)

print("Top 10 positively correlated features:")
print(target_correlation.head(11))
print("\nTop 10 negatively correlated features:")
print(target_correlation.tail(10))

# Bar chart of top correlations
top_corr = pd.concat([target_correlation.head(11), target_correlation.tail(10)])
top_corr = top_corr.drop('y_yes')

plt.figure(figsize=(14, 8))
colors = ['#5DC863' if v > 0 else '#3B528B' for v in top_corr.values]
plt.barh(top_corr.index, top_corr.values, color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Top Correlations with Term Deposit Subscription')
plt.xlabel('Pearson Correlation Coefficient')
plt.tight_layout()
plt.show()

# Focused heatmap — key features only
plt.figure(figsize=(14, 10))
key_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous',
            'y_yes', 'housing_yes', 'loan_yes', 'default_yes',
            'poutcome_success', 'poutcome_failure', 'contact_telephone']
key_cols = [c for c in key_cols if c in data_encoded.columns]
sns.heatmap(data_encoded[key_cols].corr(), annot=True, fmt='.2f',
            cmap='viridis', linewidths=0.5)
plt.title('Correlation Matrix — Key Features vs Term Deposit Subscription')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# 📊 Key Findings Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📊 KEY FINDINGS SUMMARY")
print("="*60)
print(f"Total clients           : {len(data):,}")
print(f"Age range               : {data['age'].min()} – {data['age'].max()} (mean: {data['age'].mean():.1f})")
print(f"Most common job         : {data['job'].value_counts().index[0]}")
print(f"Credit default rate     : {(data['default']=='yes').sum()/len(data)*100:.2f}%")
print(f"Mean yearly balance     : €{data['balance'].mean():,.0f}")
print(f"Clients with housing loan: {(data['housing']=='yes').sum()/len(data)*100:.1f}%")
print(f"Subscription rate       : {(data['y']=='yes').sum()/len(data)*100:.1f}%")
print(f"Strongest predictor     : duration (call length, r = 0.394)")
print(f"2nd strongest predictor : poutcome_success (r = 0.307)")
print("="*60)
