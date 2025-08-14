# Fintech Intelligence Platform: A SQL & ML System for Analytics and Fraud Detection

**Author: Srajit Bhardwaj**

## 1. Project Overview

This project showcases an end-to-end data platform designed for a digital payments ecosystem. It is composed of two primary, interconnected components:

1.  **A SQL-Based Business Intelligence Engine:** A suite of advanced, performance-optimized SQL queries designed to run directly on the database for real-time dashboards, merchant analytics, and high-level business insights.
2.  **An ML-Powered Fraud Detection Module:** A sophisticated Python-based system that uses an ensemble of machine learning and graph analytics models to detect complex, non-obvious fraud patterns.

Together, these components demonstrate a full-stack approach to data analytics—from high-speed database reporting to deep, predictive modeling.

## 2. Project Architecture

The platform operates in two stages:

1.  **At-Scale Analytics (SQL):** The database layer provides immediate, crucial business metrics for stakeholders. This is the first line of analysis for live monitoring.
2.  **Deep-Dive Modeling (Python/ML):** Data, potentially aggregated or flagged by the SQL engine, is fed into the ML module for granular risk analysis and prediction.

## 3. Technology Stack

This project utilizes a range of technologies to cover database analytics, machine learning, and data manipulation.

-   **Database & Querying:**
    -   **Advanced SQL (MySQL compatible):** Leveraged for creating complex views, Common Table Expressions (CTEs), and window functions for real-time analytics.

-   **Backend & Modeling Language:**
    -   **Python:** Used for all machine learning, feature engineering, and predictive modeling tasks.

-   **Core Data Science Libraries:**
    -   **Pandas & NumPy:** For efficient data manipulation, cleaning, and preparation.
    -   **Scikit-learn:** For implementing machine learning models, including `IsolationForest`, `RandomForestClassifier`, and `DBSCAN`.
    -   **NetworkX:** For building, analyzing, and visualizing the transaction graph to detect fraud networks.

-   **Development Environment:**
    -   **Jupyter Notebooks / VS Code:** For interactive development, analysis, and code authoring.

---

## 4. Part 1: The SQL Analytics Engine

This component is designed to live inside the database and power live dashboards for business users.

### Key Features:
-   **Market Intelligence:** Real-time views to compare market share against competitors.
-   **Merchant Analytics:** Dashboards to monitor merchant health, calculating revenue, week-over-week growth, and churn risk.
-   **User Segmentation & CLV:** Queries that segment users and calculate their predicted Customer Lifetime Value (CLV).
-   **Performance Monitoring:** Real-time health dashboards to monitor system-wide KPIs.

### Sample SQL Query: Graph-based Fraud Network Detection
```sql
-- Identifies suspicious networks based on high fraud rates and low user diversity
WITH suspicious_patterns AS (
    SELECT
        sender_bank,
        receiver_bank,
        COUNT(*) as connection_count,
        SUM(CASE WHEN fraud_flag = 1 THEN 1 ELSE 0 END) as fraud_count,
        ROUND(SUM(CASE WHEN fraud_flag = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as fraud_rate,
        COUNT(DISTINCT sender_age_group) as age_group_diversity
    FROM transactions
    WHERE timestamp >= CURDATE() - INTERVAL 7 DAY
    GROUP BY sender_bank, receiver_bank
    HAVING connection_count >= 10
),
fraud_networks AS (
    SELECT
        *,
        CASE
            WHEN fraud_rate > 5 AND age_group_diversity <= 2 THEN 'SUSPECTED_NETWORK'
            WHEN fraud_rate > 10 THEN 'HIGH_RISK_NETWORK'
            ELSE 'NORMAL_PATTERN'
        END as network_classification
    FROM suspicious_patterns
)
SELECT * FROM fraud_networks
WHERE network_classification != 'NORMAL_PATTERN'
ORDER BY fraud_rate DESC;
