# Smart Retail Analytics


This project simulates realistic customer transactions and analyzes customer behavior using RFM (Recency, Frequency, Monetary) analysis. It identifies high-risk customers for churn, segments customers into meaningful groups, and generates actionable business insights.

The system is built using custom Python classes without relying on external libraries like pandas, demonstrating advanced object-oriented programming techniques.

## 🗂 Project Structure

The code is organized into multiple files for better maintainability:

Smart-Retail-Analytics/

   │

   ├── main.py                

   ├── rfm_analyzer.py        
   
   ├── decision_tree.py        

   ├── utils.py                

   ├── requirements.txt       

   └── README.md              

## ⚙ Features

* Generate a realistic product catalog with multiple categories.

* Create a custom customer database with behavioral attributes and churn risk modeling.

* Simulate realistic transaction history over a configurable time period.

* Calculate RFM metrics and segment customers based on scores.

* Identify high-risk customers with potential churn.

* Generate comprehensive business insights and recommendations.

## 🛠 How It Works

* Setup Core Systems
* Initialize the product catalog and customer database.

* Create Customers
* Generate customers with realistic profiles and preferences.

* Simulate Transactions
* Simulate daily transactions over a specified period (e.g., 1 year).

* RFM Analysis
* Compute Recency, Frequency, and Monetary metrics for all customers.

* Customer Segmentation
* Segment customers into groups like Champions, Loyal, At-Risk, etc.

* High-Risk Customer Identification
* Identify customers with high churn propensity.

* Business Insights & Recommendations
* Generate actionable reports to optimize revenue, engagement, and retention.

## 📦 How to Run

* Clone the repository:

   git clone <repository_url>
   cd <repository_folder>


* Install dependencies:

   pip install -r requirements.txt


* Run the main script:

   python main.py

The system will simulate transactions, perform RFM analysis, and print comprehensive insights in the console.

The project is implemented without pandas or external data libraries for educational purposes.

Modular structure allows easy extension for decision tree or ML-based churn prediction.

RFM scoring logic can be customized based on business rules.
