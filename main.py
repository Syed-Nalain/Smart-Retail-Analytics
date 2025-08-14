import random
import math
from datetime import datetime, timedelta
from collections import defaultdict

class TransactionalDataFrame:
    """Custom DataFrame implementation without pandas - stores and manages tabular data"""

    def __init__(self, data=None):
        if data is None:
            self.data = []
            self.columns = []
        else:
            self.data = data
            self.columns = list(data[0].keys()) if data else []

    def add_row(self, row_dict):
        """Add a new row to the dataframe"""
        if not self.columns:
            self.columns = list(row_dict.keys())
        self.data.append(row_dict)

    def filter(self, condition_func):
        """Filter rows based on a condition function"""
        filtered_data = [row for row in self.data if condition_func(row)]
        result = TransactionalDataFrame()
        result.data = filtered_data
        result.columns = self.columns
        return result

    def group_by(self, column, aggregation_funcs):
        """Group by a column and apply aggregation functions"""
        groups = defaultdict(list)
        for row in self.data:
            groups[row[column]].append(row)

        result_data = []
        for group_key, group_rows in groups.items():
            result_row = {column: group_key}
            for agg_column, agg_func in aggregation_funcs.items():
                values = [row[agg_column] for row in group_rows if agg_column in row]
                result_row[f"{agg_column}_{agg_func.__name__}"] = agg_func(values) if values else 0
            result_data.append(result_row)

        result = TransactionalDataFrame()
        result.data = result_data
        result.columns = list(result_data[0].keys()) if result_data else []
        return result

    def get_column(self, column_name):
        """Get all values from a specific column"""
        return [row[column_name] for row in self.data if column_name in row]

    def sort_by(self, column, reverse=False):
        """Sort dataframe by a column"""
        self.data.sort(key=lambda x: x.get(column, 0), reverse=reverse)

    def head(self, n=5):
        """Return first n rows"""
        return self.data[:n]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        if not self.data:
            return "Empty TransactionalDataFrame"

        # Simple table representation
        lines = [" | ".join(self.columns)]
        lines.append("-" * len(lines[0]))
        for row in self.data[:5]:  # Show first 5 rows
            line = " | ".join(str(row.get(col, '')) for col in self.columns)
            lines.append(line)
        if len(self.data) > 5:
            lines.append(f"... ({len(self.data) - 5} more rows)")
        return "\n".join(lines)


class Product:
    """Represents a product in our catalog with realistic attributes"""

    def __init__(self, product_id, name, category, price, profit_margin=0.3):
        self.product_id = product_id
        self.name = name
        self.category = category
        self.price = price
        self.profit_margin = profit_margin
        self.profit_per_unit = price * profit_margin

    def __str__(self):
        return f"{self.name} ({self.category}) - ${self.price:.2f}"

    def __repr__(self):
        return f"Product(id={self.product_id}, name='{self.name}', price=${self.price})"


class ProductCatalog:
    """Manages our complete product inventory"""

    def __init__(self):
        self.products = {}
        self.categories = set()
        self._create_realistic_catalog()

    def _create_realistic_catalog(self):
        """Create a realistic product catalog with various categories"""
        catalog_data = [
            # Electronics
            ("Wireless Headphones", "Electronics", 89.99),
            ("Smartphone Case", "Electronics", 24.99),
            ("Laptop Charger", "Electronics", 45.99),
            ("Bluetooth Speaker", "Electronics", 129.99),
            ("Power Bank", "Electronics", 39.99),

            # Clothing
            ("Cotton T-Shirt", "Clothing", 19.99),
            ("Denim Jeans", "Clothing", 79.99),
            ("Winter Jacket", "Clothing", 159.99),
            ("Running Shoes", "Clothing", 119.99),
            ("Baseball Cap", "Clothing", 24.99),

            # Home & Garden
            ("Coffee Maker", "Home & Garden", 89.99),
            ("Throw Pillow", "Home & Garden", 29.99),
            ("LED Desk Lamp", "Home & Garden", 49.99),
            ("Plant Pot Set", "Home & Garden", 34.99),
            ("Kitchen Knife Set", "Home & Garden", 99.99),

            # Books
            ("Business Strategy Book", "Books", 29.99),
            ("Cookbook", "Books", 24.99),
            ("Fiction Novel", "Books", 14.99),
            ("Self-Help Guide", "Books", 19.99),
            ("Technical Manual", "Books", 59.99),

            # Health & Beauty
            ("Face Moisturizer", "Health & Beauty", 34.99),
            ("Vitamin Supplements", "Health & Beauty", 49.99),
            ("Yoga Mat", "Health & Beauty", 39.99),
            ("Hair Dryer", "Health & Beauty", 79.99),
            ("Essential Oils Set", "Health & Beauty", 44.99),
        ]

        for i, (name, category, price) in enumerate(catalog_data, 1):
            product = Product(i, name, category, price)
            self.products[i] = product
            self.categories.add(category)

    def get_product(self, product_id):
        """Get a specific product by ID"""
        return self.products.get(product_id)

    def get_products_by_category(self, category):
        """Get all products in a specific category"""
        return [product for product in self.products.values() if product.category == category]

    def get_random_products(self, count=1, preferred_categories=None):
        """Get random products, optionally filtering by preferred categories"""
        if preferred_categories:
            available_products = []
            for category in preferred_categories:
                available_products.extend(self.get_products_by_category(category))
        else:
            available_products = list(self.products.values())

        if not available_products:
            available_products = list(self.products.values())

        return random.sample(available_products, min(count, len(available_products)))


class Transaction:
    """Represents a single transaction with multiple products and calculated profit"""

    def __init__(self, transaction_id, customer_id, transaction_date):
        self.transaction_id = transaction_id
        self.customer_id = customer_id
        self.transaction_date = transaction_date
        self.products = []  # List of Product objects
        self.quantities = {}  # Product ID -> quantity
        self.total_amount = 0.0
        self.total_profit = 0.0

    def add_product(self, product, quantity=1):
        """Funation to add a product to this transaction"""
        self.products.append(product)
        self.quantities[product.product_id] = self.quantities.get(product.product_id, 0) + quantity
        self.total_amount += product.price * quantity
        self.total_profit += product.profit_per_unit * quantity

    def get_product_summary(self):
        """Human-readable summary of products in this transaction"""
        if not self.products:
            return "No products"

        summary = []
        for product in self.products:
            qty = self.quantities[product.product_id]
            summary.append(f"{qty}x {product.name}")
        return ", ".join(summary)

    def __str__(self):
        return f"Transaction {self.transaction_id}: ${self.total_amount:.2f} ({len(self.products)} items)"


class Customer:
    """Enhanced customer class with acquisition date and category preferences"""

    def __init__(self, customer_id, name, acquisition_date=None):
        self.customer_id = customer_id
        self.name = name
        self.acquisition_date = acquisition_date or self._random_acquisition_date()

        # Customer behavioral attributes
        self.preferred_categories = self._generate_category_preferences()
        self.avg_purchase_frequency = random.randint(1, 12)  # purchases per year
        self.avg_order_value = random.uniform(50, 300)
        self.price_sensitivity = random.uniform(0.1, 0.9)  # 0 = very price sensitive, 1 = not sensitive

        # Churn-related attributes
        self.is_churned = False
        self.churn_risk_score = 0.0  # Will be calculated based on behavior
        self.days_since_last_purchase = 0
        self.engagement_decline_rate = random.uniform(0.05, 0.15)  # How fast they disengage

    def _random_acquisition_date(self):
        """Generate a random acquisition date within the last 3 years"""
        days_ago = random.randint(30, 1095)  # 1 month to 3 years ago
        return datetime.now() - timedelta(days=days_ago)

    def _generate_category_preferences(self):
        """Generate realistic category preferences for this customer"""
        all_categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Health & Beauty"]

        # Each customer prefers 1-3 categories
        num_preferences = random.randint(1, 3)
        return random.sample(all_categories, num_preferences)

    def calculate_churn_propensity(self, days_since_last_purchase):
        """Calculate churn risk based on inactivity - gradual disengagement logic"""
        self.days_since_last_purchase = days_since_last_purchase

        # Base churn risk increases with inactivity
        inactivity_factor = min(days_since_last_purchase / 365.0, 1.0)  # Max 1.0 after a year

        # Customer tenure factor (newer customers churn faster)
        days_since_acquisition = (datetime.now() - self.acquisition_date).days
        tenure_factor = max(0.1, 1.0 - (days_since_acquisition / 1095.0))  # 3 years to stabilize

        # Engagement decline (some customers naturally disengage over time)
        decline_factor = self.engagement_decline_rate * (days_since_last_purchase / 30.0)

        # Combined churn propensity score (0 to 1)
        self.churn_risk_score = min(0.95, inactivity_factor * tenure_factor + decline_factor)

        # Mark as churned if risk is very high
        if self.churn_risk_score > 0.7:
            self.is_churned = True

        return self.churn_risk_score

    def get_purchase_likelihood(self, days_since_last_purchase):
        """Calculate likelihood of making a purchase today"""
        base_likelihood = 1.0 / (self.avg_purchase_frequency / 365.0)  # Daily likelihood

        # Decrease likelihood as time since last purchase increases
        time_factor = max(0.1, 1.0 - (days_since_last_purchase / 180.0))

        # Churn affects purchase likelihood
        churn_factor = 1.0 - self.churn_risk_score

        return base_likelihood * time_factor * churn_factor

    def __str__(self):
        return f"Customer {self.customer_id}: {self.name} (acquired: {self.acquisition_date.strftime('%Y-%m-%d')})"


class CustomerDatabase:
    """Manages our customer database with realistic data generation"""

    def __init__(self):
        self.customers = {}
        self.customer_names = self._generate_realistic_names()

    def _generate_realistic_names(self):
        """Generate a pool of realistic customer names"""
        first_names = [
            "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason", "Isabella", "William",
            "Mia", "James", "Charlotte", "Benjamin", "Amelia", "Lucas", "Harper", "Henry", "Evelyn", "Alexander",
            "Abigail", "Michael", "Emily", "Daniel", "Elizabeth", "Matthew", "Mila", "Aiden", "Ella", "Jackson",
            "Grace", "Sebastian", "Victoria", "David", "Aria", "Owen", "Scarlett", "Joseph", "Chloe", "Samuel",
            "Zoey", "John", "Luna", "Carter", "Lily", "Wyatt", "Eleanor", "Jack", "Hannah", "Luke"
        ]

        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
            "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
            "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
            "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
            "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts"
        ]

        # Generate 500 unique name combinations
        names = []
        for _ in range(500):
            first = random.choice(first_names)
            last = random.choice(last_names)
            names.append(f"{first} {last}")

        return list(set(names))  # Remove duplicates

    def create_customers(self, count):
        """Create a specified number of customers with realistic attributes"""
        print(f" Creating {count} customers with realistic profiles...")

        for i in range(1, count + 1):
            name = random.choice(self.customer_names)
            customer = Customer(i, name)
            self.customers[i] = customer

        print(f"✅ Created {len(self.customers)} customers")
        print(f"   Sample customer: {list(self.customers.values())[0]}")

    def get_customer(self, customer_id):
        """Get a customer by ID"""
        return self.customers.get(customer_id)

    def get_all_customers(self):
        """Get all customers"""
        return list(self.customers.values())

    def get_churned_customers(self):
        """Get customers who have churned"""
        return [customer for customer in self.customers.values() if customer.is_churned]

    def get_active_customers(self):
        """Get active (non-churned) customers"""
        return [customer for customer in self.customers.values() if not customer.is_churned]

    def __str__(self):
        return f"Customer {self.customer_id}: {self.name} (acquired: {self.acquisition_date.strftime('%Y-%m-%d')})"
class TransactionSimulator:
    """Simulates realistic transaction history with gradual churn logic"""

    def __init__(self, customer_db, product_catalog):
        self.customer_db = customer_db
        self.product_catalog = product_catalog
        self.transactions = TransactionalDataFrame()
        self.transaction_counter = 1

    def simulate_transaction_history(self, days=365):
        """Simulate realistic transaction history over specified period"""
        print(f" Simulating {days} days of transaction history...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        transaction_data = []

        # Simulate day by day
        for day_offset in range(days):
            current_date = start_date + timedelta(days=day_offset)
            days_from_start = day_offset

            # Check each customer for potential purchases
            for customer in self.customer_db.get_all_customers():
                # Calculate days since customer's last purchase
                days_since_last = self._calculate_days_since_last_purchase(customer.customer_id, current_date)

                # Update customer's churn propensity
                customer.calculate_churn_propensity(days_since_last)

                # Check if customer makes a purchase today
                purchase_likelihood = customer.get_purchase_likelihood(days_since_last)

                if random.random() < purchase_likelihood:
                    transaction = self._create_realistic_transaction(customer, current_date)
                    if transaction:
                        # Convert transaction to dictionary for our custom DataFrame
                        transaction_dict = {
                            'transaction_id': transaction.transaction_id,
                            'customer_id': transaction.customer_id,
                            'date': transaction.transaction_date,
                            'total_amount': transaction.total_amount,
                            'total_profit': transaction.total_profit,
                            'product_count': len(transaction.products),
                            'products': transaction.get_product_summary()
                        }
                        transaction_data.append(transaction_dict)

        # Store in our custom DataFrame
        self.transactions = TransactionalDataFrame(transaction_data)

        print(f"✅ Generated {len(self.transactions)} transactions")
        print(f"   Average transaction value: ${self._calculate_average_transaction_value():.2f}")

        return self.transactions

    def _calculate_days_since_last_purchase(self, customer_id, current_date):
        """Calculate days since customer's last purchase"""
        customer_transactions = self.transactions.filter(
            lambda row: row['customer_id'] == customer_id
        )

        if len(customer_transactions) == 0:
            # No previous purchases - use acquisition date
            customer = self.customer_db.get_customer(customer_id)
            return (current_date - customer.acquisition_date).days

        # Find most recent transaction
        last_transaction_date = max(
            [datetime.fromisoformat(row['date'].isoformat()) for row in customer_transactions.data]
        )

        return (current_date - last_transaction_date).days

    def _create_realistic_transaction(self, customer, transaction_date):
        """Create a realistic transaction for a customer"""
        transaction = Transaction(self.transaction_counter, customer.customer_id, transaction_date)
        self.transaction_counter += 1

        # Determine number of products (most transactions have 1-3 products)
        num_products = random.choices([1, 2, 3, 4], weights=[50, 30, 15, 5])[0]

        # Select products based on customer preferences
        selected_products = self.product_catalog.get_random_products(
            count=num_products,
            preferred_categories=customer.preferred_categories
        )

        for product in selected_products:
            # Quantity usually 1, sometimes 2
            quantity = random.choices([1, 2], weights=[80, 20])[0]
            transaction.add_product(product, quantity)

        return transaction

    def _calculate_average_transaction_value(self):
        """Calculate average transaction value"""
        if len(self.transactions) == 0:
            return 0.0

        total_value = sum(row['total_amount'] for row in self.transactions.data)
        return total_value / len(self.transactions)

    def get_transactions(self):
        """Get the transaction DataFrame"""
        return self.transactions


class RFMAnalyzer:
    """Analyzes customer behavior using RFM (Recency, Frequency, Monetary) analysis"""

    def __init__(self, customer_db, transactions):
        self.customer_db = customer_db
        self.transactions = transactions
        self.rfm_data = TransactionalDataFrame()

    def calculate_rfm_metrics(self):
        """Calculate RFM metrics for all customers"""
        print(" Calculating RFM (Recency, Frequency, Monetary) metrics...")

        # Get reference date (most recent transaction)
        all_dates = [datetime.fromisoformat(row['date'].isoformat()) for row in self.transactions.data]
        reference_date = max(all_dates) if all_dates else datetime.now()

        rfm_data = []

        for customer in self.customer_db.get_all_customers():
            # Get customer's transactions
            customer_transactions = self.transactions.filter(
                lambda row: row['customer_id'] == customer.customer_id
            )

            if len(customer_transactions) == 0:
                # Customer has no transactions
                rfm_data.append({
                    'customer_id': customer.customer_id,
                    'customer_name': customer.name,
                    'recency': (reference_date - customer.acquisition_date).days,
                    'frequency': 0,
                    'monetary': 0.0,
                    'is_churned': customer.is_churned,
                    'churn_risk': customer.churn_risk_score,
                    'preferred_categories': ', '.join(customer.preferred_categories)
                })
                continue

            # Calculate metrics
            transaction_dates = [datetime.fromisoformat(row['date'].isoformat()) for row in customer_transactions.data]
            amounts = [row['total_amount'] for row in customer_transactions.data]

            recency = (reference_date - max(transaction_dates)).days
            frequency = len(customer_transactions)
            monetary = sum(amounts)

            rfm_data.append({
                'customer_id': customer.customer_id,
                'customer_name': customer.name,
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary,
                'is_churned': customer.is_churned,
                'churn_risk': customer.churn_risk_score,
                'preferred_categories': ', '.join(customer.preferred_categories)
            })

        self.rfm_data = TransactionalDataFrame(rfm_data)

        print(f"✅ Calculated RFM metrics for {len(self.rfm_data)} customers")
        self._print_rfm_summary()

        return self.rfm_data

    def _print_rfm_summary(self):
        """Print a summary of RFM metrics"""
        if len(self.rfm_data) == 0:
            return

        recencies = [row['recency'] for row in self.rfm_data.data]
        frequencies = [row['frequency'] for row in self.rfm_data.data]
        monetaries = [row['monetary'] for row in self.rfm_data.data]
        churn_rates = [row['is_churned'] for row in self.rfm_data.data]

        print(f"   Average Recency: {sum(recencies) / len(recencies):.1f} days")
        print(f"   Average Frequency: {sum(frequencies) / len(frequencies):.1f} transactions")
        print(f"   Average Monetary: ${sum(monetaries) / len(monetaries):.2f}")
        print(f"   Overall Churn Rate: {sum(churn_rates) / len(churn_rates):.2%}")

    def segment_customers(self):
        """Segment customers into different groups based on RFM scores"""
        print(" Segmenting customers based on RFM analysis...")

        if len(self.rfm_data) == 0:
            print("❌ No RFM data available. Run calculate_rfm_metrics first.")
            return

        # Calculate RFM scores (1-5 scale)
        recencies = [row['recency'] for row in self.rfm_data.data]
        frequencies = [row['frequency'] for row in self.rfm_data.data]
        monetaries = [row['monetary'] for row in self.rfm_data.data]

        # Calculate quartiles for scoring
        def calculate_score(value, values_list, reverse=False):
            if not values_list or max(values_list) == min(values_list):
                return 3  # Default middle score

            sorted_values = sorted(values_list)
            n = len(sorted_values)

            if value <= sorted_values[n // 4]:
                return 5 if reverse else 1
            elif value <= sorted_values[n // 2]:
                return 4 if reverse else 2
            elif value <= sorted_values[3 * n // 4]:
                return 3
            else:
                return 1 if reverse else 5

        # Add scores and segments to data
        for row in self.rfm_data.data:
            r_score = calculate_score(row['recency'], recencies, reverse=True)  # Lower recency = higher score
            f_score = calculate_score(row['frequency'], frequencies)
            m_score = calculate_score(row['monetary'], monetaries)

            row['r_score'] = r_score
            row['f_score'] = f_score
            row['m_score'] = m_score
            row['rfm_score'] = f"{r_score}{f_score}{m_score}"

            # Assign segment based on RFM combination
            if r_score >= 4 and f_score >= 4:
                segment = "Champions"
            elif r_score >= 4 and f_score >= 2:
                segment = "Loyal Customers"
            elif r_score >= 3 and f_score >= 3 and m_score >= 3:
                segment = "Potential Loyalists"
            elif r_score >= 4 and f_score <= 2:
                segment = "New Customers"
            elif r_score <= 2 and f_score >= 3:
                segment = "At Risk"
            elif r_score <= 2 and f_score <= 2:
                segment = "Cannot Lose Them" if m_score >= 4 else "Lost"
            else:
                segment = "Need Attention"

            row['segment'] = segment

        self._print_segment_analysis()

    def _print_segment_analysis(self):
        """Print analysis of customer segments"""
        print("\n=== CUSTOMER SEGMENT ANALYSIS ===")

        # Group by segment
        segments = {}
        for row in self.rfm_data.data:
            segment = row['segment']
            if segment not in segments:
                segments[segment] = []
            segments[segment].append(row)

        for segment_name, customers in segments.items():
            churn_rate = sum(1 for c in customers if c['is_churned']) / len(customers)
            avg_recency = sum(c['recency'] for c in customers) / len(customers)
            avg_frequency = sum(c['frequency'] for c in customers) / len(customers)
            avg_monetary = sum(c['monetary'] for c in customers) / len(customers)
            avg_churn_risk = sum(c['churn_risk'] for c in customers) / len(customers)

            print(f"\n {segment_name} ({len(customers)} customers):")
            print(f"   Churn Rate: {churn_rate:.1%}")
            print(f"   Avg Churn Risk: {avg_churn_risk:.2f}")
            print(f"   Avg Recency: {avg_recency:.0f} days")
            print(f"   Avg Frequency: {avg_frequency:.1f} purchases")
            print(f"   Avg Monetary: ${avg_monetary:.2f}")

    def get_high_risk_customers(self, risk_threshold=0.6):
        """Get customers with high churn risk"""
        high_risk = self.rfm_data.filter(
            lambda row: row['churn_risk'] >= risk_threshold
        )
        return high_risk

    def get_rfm_data(self):
        """Get the RFM analysis results"""
        return self.rfm_data


def main():
    """Main execution - orchestrates the entire customer analysis pipeline"""
    print("Starting Enhanced Customer Churn Analysis System")
    print("=" * 60)

    # 1. Initialize core systems
    print("\n1️⃣ Setting up core systems...")
    product_catalog = ProductCatalog()
    customer_db = CustomerDatabase()

    print(
        f" Product catalog: {len(product_catalog.products)} products across {len(product_catalog.categories)} categories")
    print(f"   Categories: {', '.join(product_catalog.categories)}")

    # 2. Create customers
    print("\n2️⃣ Creating customer base...")
    customer_db.create_customers(10)  # Create 200 realistic customers
    #print(customer_db.get_all_customers())
    for i in range(1, 200 + 1):
        print(customer_db.get_customer(i))
    # 3. Simulate transactions
    print("\n3️⃣ Simulating transaction history...")
    transaction_sim = TransactionSimulator(customer_db, product_catalog)
    transactions = transaction_sim.simulate_transaction_history(365)  # 1 year of data
    print(transactions.head(30))

    # 4. Analyze customer behavior
    print("\n4️⃣ Analyzing customer behavior with RFM...")
    rfm_analyzer = RFMAnalyzer(customer_db, transactions)
    rfm_data = rfm_analyzer.calculate_rfm_metrics()
    rfm_analyzer.segment_customers()

    # 5. Identify at-risk customers
    print("\n5️⃣ Identifying high-risk customers...")
    high_risk_customers = rfm_analyzer.get_high_risk_customers(0.5)

    if len(high_risk_customers) > 0:
        print(f"  Found {len(high_risk_customers)} customers at high risk of churning:")
        for customer_data in high_risk_customers.head(5):  # Show top 5
            print(f"   • {customer_data['customer_name']} (Risk: {customer_data['churn_risk']:.2f}, "
                  f"Recency: {customer_data['recency']} days)")

    # 6. Generate comprehensive business insights
    print("\n6️⃣ Generating business insights...")
    insights_generator = BusinessInsightsGenerator(customer_db, transactions, rfm_data)
    insights_generator.generate_comprehensive_report()

    # 7. Summary insights
    print("\n7️⃣ Final Summary:")
    active_customers = customer_db.get_active_customers()
    churned_customers = customer_db.get_churned_customers()

    print(f"    Total Customers: {len(customer_db.customers)}")
    print(f"    Active: {len(active_customers)} ({len(active_customers) / len(customer_db.customers) * 100:.1f}%)")
    print(f"    Churned: {len(churned_customers)} ({len(churned_customers) / len(customer_db.customers) * 100:.1f}%)")

    # Calculate total business metrics
    total_revenue = sum(row['total_amount'] for row in transactions.data)
    total_profit = sum(row['total_profit'] for row in transactions.data)

    print(f"    Total Revenue: ${total_revenue:.2f}")
    print(f"    Total Profit: ${total_profit:.2f} ({total_profit / total_revenue * 100:.1f}% margin)")

    # Show sample data
    print("\n8️⃣ Sample Data Preview:")
    print("\nSample RFM Analysis Results:")
    print(rfm_analyzer.get_rfm_data())
 #   transactions.get_transactions()
 #  print(customer_db.get_all_customers())



class BusinessInsightsGenerator:
    """Generate actionable business insights from the analysis"""

    def __init__(self, customer_db, transactions, rfm_data):
        self.customer_db = customer_db
        self.transactions = transactions
        self.rfm_data = rfm_data

    def generate_comprehensive_report(self):
        """Generate a comprehensive business report"""
        print("\n COMPREHENSIVE BUSINESS INSIGHTS REPORT")
        print("=" * 60)

        self._analyze_revenue_trends()
        self._analyze_customer_segments()
        self._analyze_product_performance()
        self._generate_recommendations()

    def _analyze_revenue_trends(self):
        """Analyze revenue and profit trends"""
        print("\n REVENUE ANALYSIS")
        print("-" * 30)

        if len(self.transactions) == 0:
            print("No transaction data available.")
            return

        total_revenue = sum(row['total_amount'] for row in self.transactions.data)
        total_profit = sum(row['total_profit'] for row in self.transactions.data)
        avg_order_value = total_revenue / len(self.transactions)

        print(f"Total Revenue: ${total_revenue:,.2f}")
        print(f"Total Profit: ${total_profit:,.2f}")
        print(f"Profit Margin: {(total_profit / total_revenue) * 100:.1f}%")
        print(f"Average Order Value: ${avg_order_value:.2f}")
        print(f"Total Transactions: {len(self.transactions):,}")

    def _analyze_customer_segments(self):
        """Analyze customer segment performance"""
        print("\n CUSTOMER SEGMENT PERFORMANCE")
        print("-" * 35)

        segments = {}
        for row in self.rfm_data.data:
            segment = row.get('segment', 'Unknown')
            if segment not in segments:
                segments[segment] = {
                    'customers': [],
                    'total_revenue': 0,
                    'churn_count': 0
                }

            segments[segment]['customers'].append(row)
            segments[segment]['total_revenue'] += row['monetary']
            if row['is_churned']:
                segments[segment]['churn_count'] += 1

        for segment_name, data in segments.items():
            customer_count = len(data['customers'])
            avg_revenue = data['total_revenue'] / customer_count if customer_count > 0 else 0
            churn_rate = data['churn_count'] / customer_count if customer_count > 0 else 0

            print(f"\n{segment_name}:")
            print(f"  Customers: {customer_count}")
            print(f"  Avg Revenue per Customer: ${avg_revenue:.2f}")
            print(f"  Churn Rate: {churn_rate:.1%}")
            print(f"  Total Segment Revenue: ${data['total_revenue']:,.2f}")

    def _analyze_product_performance(self):
        """Analyze which product categories are performing best"""
        print("\n PRODUCT CATEGORY INSIGHTS")
        print("-" * 32)

        category_performance = {}

        for row in self.transactions.data:
            # Extract category info from customer preferences (simplified)
            customer = self.customer_db.get_customer(row['customer_id'])
            for category in customer.preferred_categories:
                if category not in category_performance:
                    category_performance[category] = {
                        'revenue': 0,
                        'transactions': 0
                    }

                # Rough allocation based on preference
                category_performance[category]['revenue'] += row['total_amount'] / len(customer.preferred_categories)
                category_performance[category]['transactions'] += 1 / len(customer.preferred_categories)

        # Sort by revenue
        sorted_categories = sorted(
            category_performance.items(),
            key=lambda x: x[1]['revenue'],
            reverse=True
        )

        for category, data in sorted_categories:
            avg_transaction = data['revenue'] / data['transactions'] if data['transactions'] > 0 else 0
            print(f"{category}:")
            print(f"  Revenue: ${data['revenue']:,.2f}")
            print(f"  Transactions: {data['transactions']:.0f}")
            print(f"  Avg per Transaction: ${avg_transaction:.2f}")

    def _generate_recommendations(self):
        """Generate actionable business recommendations"""
        print("\n💡 STRATEGIC RECOMMENDATIONS")
        print("-" * 32)

        # Analyze churn patterns
        churned_customers = [row for row in self.rfm_data.data if row['is_churned']]
        high_risk_customers = [row for row in self.rfm_data.data if row['churn_risk'] > 0.6]

        print("1. CHURN PREVENTION:")
        if high_risk_customers:
            print(f"   • {len(high_risk_customers)} customers are at high churn risk")
            print("   • Implement targeted retention campaigns")
            print("   • Offer personalized discounts or loyalty rewards")

        print("\n2. CUSTOMER ENGAGEMENT:")
        inactive_customers = [row for row in self.rfm_data.data if row['recency'] > 90]
        print(f"   • {len(inactive_customers)} customers haven't purchased in 90+ days")
        print("   • Launch re-engagement email campaigns")
        print("   • Consider win-back offers")

        print("\n3. REVENUE OPTIMIZATION:")
        low_frequency_customers = [row for row in self.rfm_data.data if 0 < row['frequency'] <= 2]
        print(f"   • {len(low_frequency_customers)} customers have made only 1-2 purchases")
        print("   • Focus on converting them to repeat customers")
        print("   • Implement new customer nurture sequences")

        print("\n4. SEGMENT-SPECIFIC ACTIONS:")
        segments = set(row.get('segment', 'Unknown') for row in self.rfm_data.data)
        for segment in segments:
            if segment == "Champions":
                print(f"   • {segment}: Leverage for referrals and testimonials")
            elif segment == "At Risk":
                print(f"   • {segment}: Immediate intervention required")
            elif segment == "New Customers":
                print(f"   • {segment}: Focus on onboarding and first repeat purchase")
if __name__ == "__main__":
    main()
