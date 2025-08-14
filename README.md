# Smart Retail System — RFM Analysis + Decision Tree 🛒

A Python-powered retail analytics system that uses **RFM (Recency, Frequency, Monetary)** analysis and a **Decision Tree model** to automatically segment customers.  
Designed to help businesses make data-driven marketing decisions and improve customer retention.

---

## 📊 What is RFM Analysis?
RFM is a marketing technique that measures:
- **Recency** — How recently a customer purchased
- **Frequency** — How often they purchase
- **Monetary** — How much they spend

By scoring customers on these dimensions, businesses can prioritize and personalize engagement.

---

## 🛠 How It Works
1. **Data Collection** — Load transaction history (customer ID, purchase date, amount).
2. **RFM Calculation** — Compute Recency, Frequency, and Monetary values for each customer.
3. **Scoring** — Assign scores (1–5) for each metric and combine into an RFM profile.
4. **Decision Tree Classification** — Train a model to classify customers into segments like:
   - **Loyal Customer**
   - **At Risk**
   - **Churned**
   - **New Customer**
5. **Prediction & Insights** — Apply the model to new data to predict customer segments.

---

## 🧠 Machine Learning Model
- **Algorithm**: `DecisionTreeClassifier` from scikit-learn
- **Input Features**: RFM scores
- **Output Labels**: Customer segment
- **Training Process**:
  1. Load historical customer data with known segments
  2. Split into training/testing sets
  3. Train the decision tree to learn patterns in RFM scores
  4. Evaluate accuracy on test data

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Install dependencies:
```bash
pip install pandas scikit-learn
