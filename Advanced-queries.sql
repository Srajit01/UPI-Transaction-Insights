
CREATE VIEW phonepe_market_position AS
WITH monthly_volumes AS (
    SELECT 
        'PhonePe' as platform,
        DATE_FORMAT(timestamp, '%Y-%m') as month,
        COUNT(*) as transaction_count,
        SUM(amount_inr) as transaction_value
    FROM phonepe_transactions
    GROUP BY month
    
    UNION ALL

    SELECT 
        'Google Pay' as platform,
        month,
        ROUND(transaction_count * 0.75) as transaction_count,  -- 35% vs 46% market share
        ROUND(transaction_value * 0.75) as transaction_value
    FROM (SELECT DATE_FORMAT(timestamp, '%Y-%m') as month, COUNT(*) as transaction_count, SUM(amount_inr) as transaction_value FROM phonepe_transactions GROUP BY month) t
    
    UNION ALL
    
    SELECT 
        'Paytm' as platform,
        month,
        ROUND(transaction_count * 0.15) as transaction_count,  -- 7% market share
        ROUND(transaction_value * 0.15) as transaction_value
    FROM (SELECT DATE_FORMAT(timestamp, '%Y-%m') as month, COUNT(*) as transaction_count, SUM(amount_inr) as transaction_value FROM phonepe_transactions GROUP BY month) t
)
SELECT 
    platform,
    month,
    transaction_count,
    transaction_value,
    ROUND(transaction_count * 100.0 / SUM(transaction_count) OVER (PARTITION BY month), 2) as volume_market_share,
    ROUND(transaction_value * 100.0 / SUM(transaction_value) OVER (PARTITION BY month), 2) as value_market_share
FROM monthly_volumes
ORDER BY month DESC, transaction_count DESC;

SELECT 
    'PhonePe Performance' as metric_category,
    ROUND(AVG(CASE WHEN transaction_status = 'SUCCESS' THEN 1.0 ELSE 0.0 END) * 100, 2) as success_rate,
    ROUND(AVG(CASE WHEN fraud_flag = 1 THEN 1.0 ELSE 0.0 END) * 100, 4) as fraud_rate,
    ROUND(AVG(amount_inr), 2) as avg_transaction_value,
    COUNT(*) / COUNT(DISTINCT DATE(timestamp)) as daily_avg_transactions,
    CASE 
        WHEN AVG(CASE WHEN transaction_status = 'SUCCESS' THEN 1.0 ELSE 0.0 END) > 0.985 THEN 'EXCELLENT'
        WHEN AVG(CASE WHEN transaction_status = 'SUCCESS' THEN 1.0 ELSE 0.0 END) > 0.970 THEN 'GOOD'
        ELSE 'NEEDS_IMPROVEMENT'
    END as success_rate_grade,
    CASE 
        WHEN AVG(CASE WHEN fraud_flag = 1 THEN 1.0 ELSE 0.0 END) < 0.001 THEN 'EXCELLENT'
        WHEN AVG(CASE WHEN fraud_flag = 1 THEN 1.0 ELSE 0.0 END) < 0.005 THEN 'GOOD'
        ELSE 'NEEDS_IMPROVEMENT'
    END as fraud_rate_grade
FROM phonepe_transactions
WHERE timestamp >= CURDATE() - INTERVAL 30 DAY;

-- 3. Real-time Fraud Risk Scoring
CREATE VIEW realtime_fraud_monitor AS
SELECT 
    transaction_id,
    timestamp,
    amount_inr,
    sender_bank,
    receiver_bank,
    device_type,
    network_type,
    sender_state,
    
  
    CASE 
        WHEN amount_inr > 10000 THEN 2
        WHEN amount_inr > 5000 THEN 1
        ELSE 0
    END +
    CASE 
        WHEN network_type = '5G' THEN 1  
        WHEN network_type = 'WiFi' THEN 0
        ELSE 0.5
    END +
    CASE 
        WHEN HOUR(timestamp) BETWEEN 22 AND 6 THEN 1  
        ELSE 0
    END +
    CASE 
        WHEN sender_state != receiver_state THEN 1  
        ELSE 0
    END as calculated_risk_score,
    
    fraud_flag as actual_fraud,
    
    CASE 
        WHEN fraud_flag = 1 THEN 'CONFIRMED_FRAUD'
        WHEN calculated_risk_score >= 3 THEN 'HIGH_RISK'
        WHEN calculated_risk_score >= 2 THEN 'MEDIUM_RISK'
        ELSE 'LOW_RISK'
    END as risk_category
    
FROM phonepe_transactions
WHERE timestamp >= NOW() - INTERVAL 1 HOUR
ORDER BY calculated_risk_score DESC, timestamp DESC;

-- 4. Graph-based Fraud Network Detection
WITH suspicious_patterns AS (
    SELECT 
        sender_bank,
        receiver_bank,
        COUNT(*) as connection_count,
        SUM(CASE WHEN fraud_flag = 1 THEN 1 ELSE 0 END) as fraud_count,
        ROUND(SUM(CASE WHEN fraud_flag = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as fraud_rate,
        SUM(amount_inr) as total_amount,
        COUNT(DISTINCT sender_age_group) as age_group_diversity,
        COUNT(DISTINCT device_type) as device_diversity
    FROM phonepe_transactions
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
            WHEN connection_count > 1000 AND fraud_rate > 1 THEN 'VOLUME_RISK_NETWORK'
            ELSE 'NORMAL_PATTERN'
        END as network_classification
    FROM suspicious_patterns
)
SELECT * FROM fraud_networks
WHERE network_classification != 'NORMAL_PATTERN'
ORDER BY fraud_rate DESC, connection_count DESC;

-- 5. Device Fingerprinting and Anomaly Detection
SELECT 
    device_type,
    COUNT(DISTINCT transaction_id) as unique_devices,
    COUNT(*) as total_transactions,
    ROUND(COUNT(*) / COUNT(DISTINCT transaction_id), 2) as avg_transactions_per_device,
    ROUND(SUM(CASE WHEN fraud_flag = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as device_fraud_rate,
    ROUND(AVG(amount_inr), 2) as avg_transaction_amount,
    
    CASE 
        WHEN COUNT(*) / COUNT(DISTINCT transaction_id) > 50 THEN 'HIGH_VELOCITY_DEVICE'
        WHEN SUM(CASE WHEN fraud_flag = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) > 2 THEN 'HIGH_FRAUD_DEVICE'
        WHEN AVG(amount_inr) > 5000 THEN 'HIGH_VALUE_DEVICE'
        ELSE 'NORMAL_DEVICE'
    END as device_risk_profile
    
FROM phonepe_transactions
WHERE timestamp >= CURDATE() - INTERVAL 30 DAY
GROUP BY device_type
ORDER BY device_fraud_rate DESC, avg_transactions_per_device DESC;

-- 6. Merchant Performance Dashboard
CREATE VIEW merchant_intelligence AS
WITH merchant_metrics AS (
    SELECT 
        merchant_category,
        sender_state,
        COUNT(*) as transaction_volume,
        COUNT(DISTINCT DATE(timestamp)) as active_days,
        SUM(amount_inr) as total_revenue,
        AVG(amount_inr) as avg_ticket_size,
        ROUND(COUNT(*) / COUNT(DISTINCT DATE(timestamp)), 0) as daily_avg_transactions,
        ROUND(SUM(CASE WHEN transaction_status = 'SUCCESS' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as success_rate,
        ROUND(SUM(CASE WHEN fraud_flag = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) as fraud_rate,
        COUNT(DISTINCT sender_age_group) as customer_age_diversity,
        
        -- Growth metrics
        COUNT(CASE WHEN timestamp >= CURDATE() - INTERVAL 7 DAY THEN 1 END) as last_7_days_txns,
        COUNT(CASE WHEN timestamp >= CURDATE() - INTERVAL 14 DAY AND timestamp < CURDATE() - INTERVAL 7 DAY THEN 1 END) as prev_7_days_txns
        
    FROM phonepe_transactions
    WHERE timestamp >= CURDATE() - INTERVAL 30 DAY
    GROUP BY merchant_category, sender_state
),
merchant_rankings AS (
    SELECT 
        *,
        ROUND((last_7_days_txns - prev_7_days_txns) * 100.0 / NULLIF(prev_7_days_txns, 0), 2) as week_over_week_growth,
        ROW_NUMBER() OVER (PARTITION BY sender_state ORDER BY total_revenue DESC) as revenue_rank_in_state,
        ROW_NUMBER() OVER (ORDER BY total_revenue DESC) as overall_revenue_rank,
        
        -- Merchant health score
        CASE 
            WHEN success_rate >= 98 AND fraud_rate <= 0.5 AND daily_avg_transactions >= 100 THEN 'EXCELLENT'
            WHEN success_rate >= 95 AND fraud_rate <= 1 AND daily_avg_transactions >= 50 THEN 'GOOD'
            WHEN success_rate >= 90 AND fraud_rate <= 2 THEN 'AVERAGE'
            ELSE 'NEEDS_ATTENTION'
        END as merchant_health_score
        
    FROM merchant_metrics
)
SELECT * FROM merchant_rankings
ORDER BY overall_revenue_rank;

-- 7. Merchant Churn Risk Analysis
WITH merchant_activity AS (
    SELECT 
        merchant_category,
        COUNT(*) as current_month_txns,
        LAG(COUNT(*)) OVER (PARTITION BY merchant_category ORDER BY DATE_FORMAT(timestamp, '%Y-%m')) as prev_month_txns,
        DATE_FORMAT(timestamp, '%Y-%m') as month
    FROM phonepe_transactions
    WHERE timestamp >= CURDATE() - INTERVAL 60 DAY
    GROUP BY merchant_category, DATE_FORMAT(timestamp, '%Y-%m')
),
churn_indicators AS (
    SELECT 
        merchant_category,
        month,
        current_month_txns,
        prev_month_txns,
        ROUND((current_month_txns - prev_month_txns) * 100.0 / prev_month_txns, 2) as month_over_month_change,
        
        CASE 
            WHEN prev_month_txns > 0 AND current_month_txns = 0 THEN 'CHURNED'
            WHEN (current_month_txns - prev_month_txns) * 100.0 / prev_month_txns < -50 THEN 'HIGH_CHURN_RISK'
            WHEN (current_month_txns - prev_month_txns) * 100.0 / prev_month_txns < -25 THEN 'MEDIUM_CHURN_RISK'
            WHEN (current_month_txns - prev_month_txns) * 100.0 / prev_month_txns > 25 THEN 'GROWING'
            ELSE 'STABLE'
        END as churn_risk_level
        
    FROM merchant_activity
    WHERE prev_month_txns IS NOT NULL
)
SELECT * FROM churn_indicators
WHERE churn_risk_level IN ('CHURNED', 'HIGH_CHURN_RISK', 'MEDIUM_CHURN_RISK')
ORDER BY month_over_month_change;

-- 8. Advanced User Segmentation
CREATE VIEW phonepe_user_segments AS
WITH user_behavior AS (
    SELECT 
        sender_age_group,
        sender_state,
        device_type,
        network_type,
        COUNT(*) as total_transactions,
        COUNT(DISTINCT merchant_category) as merchant_diversity,
        COUNT(DISTINCT DATE(timestamp)) as active_days,
        SUM(amount_inr) as total_spent,
        AVG(amount_inr) as avg_transaction_value,
        MAX(amount_inr) as max_transaction_value,
        
        COUNT(CASE WHEN transaction_type = 'P2P' THEN 1 END) as p2p_transactions,
        COUNT(CASE WHEN transaction_type = 'P2M' THEN 1 END) as p2m_transactions,
      
        COUNT(CASE WHEN HOUR(timestamp) BETWEEN 9 AND 17 THEN 1 END) as business_hours_txns,
        COUNT(CASE WHEN HOUR(timestamp) BETWEEN 18 AND 22 THEN 1 END) as evening_txns,
        COUNT(CASE WHEN DAYOFWEEK(timestamp) IN (1,7) THEN 1 END) as weekend_txns
        
    FROM phonepe_transactions
    WHERE timestamp >= CURDATE() - INTERVAL 90 DAY
    GROUP BY sender_age_group, sender_state, device_type, network_type
),
user_classification AS (
    SELECT 
        *,
        ROUND(total_transactions / NULLIF(active_days, 0), 2) as transactions_per_active_day,
        ROUND(p2p_transactions * 100.0 / total_transactions, 1) as p2p_percentage,
        ROUND(p2m_transactions * 100.0 / total_transactions, 1) as p2m_percentage,
       
        CASE 
            WHEN total_spent > 50000 AND avg_transaction_value > 2000 THEN 'HIGH_VALUE_USER'
            WHEN total_transactions > 200 AND avg_transaction_value < 500 THEN 'FREQUENT_SMALL_SPENDER'
            WHEN p2m_transactions > p2p_transactions AND merchant_diversity > 10 THEN 'MERCHANT_POWER_USER'
            WHEN sender_age_group IN ('18-25', '26-35') AND device_type = 'Android' THEN 'DIGITAL_NATIVE'
            WHEN weekend_txns * 100.0 / total_transactions > 40 THEN 'WEEKEND_SHOPPER'
            WHEN business_hours_txns * 100.0 / total_transactions > 60 THEN 'BUSINESS_USER'
            ELSE 'REGULAR_USER'
        END as user_segment
        
    FROM user_behavior
)
SELECT 
    user_segment,
    COUNT(*) as segment_size,
    ROUND(AVG(total_spent), 2) as avg_spending,
    ROUND(AVG(total_transactions), 0) as avg_transactions,
    ROUND(AVG(avg_transaction_value), 2) as avg_ticket_size,
    ROUND(AVG(merchant_diversity), 1) as avg_merchant_diversity
FROM user_classification
GROUP BY user_segment
ORDER BY avg_spending DESC;


WITH user_metrics AS (
    SELECT 
        sender_age_group,
        sender_state,
        device_type,
        MIN(DATE(timestamp)) as first_transaction_date,
        MAX(DATE(timestamp)) as last_transaction_date,
        DATEDIFF(MAX(DATE(timestamp)), MIN(DATE(timestamp))) + 1 as customer_lifespan_days,
        COUNT(*) as total_transactions,
        SUM(amount_inr) as total_value,
        COUNT(DISTINCT merchant_category) as unique_merchants,
        COUNT(DISTINCT DATE(timestamp)) as active_days
    FROM phonepe_transactions
    WHERE timestamp >= CURDATE() - INTERVAL 365 DAY
    GROUP BY sender_age_group, sender_state, device_type
    HAVING COUNT(*) >= 10  
),
clv_calculation AS (
    SELECT 
        *,
        ROUND(total_value / NULLIF(customer_lifespan_days, 0), 2) as daily_value,
        ROUND(total_transactions / NULLIF(customer_lifespan_days, 0), 2) as daily_frequency,
        ROUND(total_value / NULLIF(total_transactions, 0), 2) as avg_order_value,
        ROUND(active_days / NULLIF(customer_lifespan_days, 0) * 100, 1) as engagement_rate,
        
        
        ROUND(
            (total_value / NULLIF(customer_lifespan_days, 0)) * 365 * 
            CASE 
                WHEN active_days / NULLIF(customer_lifespan_days, 0) > 0.5 THEN 2  
                WHEN active_days / NULLIF(customer_lifespan_days, 0) > 0.2 THEN 1.5
                ELSE 1
            END, 2
        ) as predicted_annual_clv
        
    FROM user_metrics
)
SELECT 
    sender_age_group,
    sender_state,
    device_type,
    COUNT(*) as user_count,
    ROUND(AVG(predicted_annual_clv), 2) as avg_predicted_clv,
    ROUND(AVG(total_value), 2) as avg_historical_value,
    ROUND(AVG(engagement_rate), 1) as avg_engagement_rate,
    ROUND(AVG(unique_merchants), 1) as avg_merchant_diversity
FROM clv_calculation
GROUP BY sender_age_group, sender_state, device_type
ORDER BY avg_predicted_clv DESC;

-- =====================================
-- REAL-TIME MONITORING QUERIES
-- =====================================

-- 10. Real-time Transaction Health Monitor
CREATE VIEW realtime_health_dashboard AS
SELECT 
    'LAST_HOUR' as time_period,
    COUNT(*) as transaction_count,
    SUM(amount_inr) as total_value,
    ROUND(AVG(amount_inr), 2) as avg_transaction_value,
    ROUND(COUNT(CASE WHEN transaction_status = 'SUCCESS' THEN 1 END) * 100.0 / COUNT(*), 2) as success_rate,
    ROUND(COUNT(CASE WHEN fraud_flag = 1 THEN 1 END) * 100.0 / COUNT(*), 4) as fraud_rate,
    COUNT(DISTINCT sender_bank) as active_banks,
    COUNT(DISTINCT merchant_category) as active_merchant_categories,
    
    -- Alerts
    CASE 
        WHEN COUNT(CASE WHEN transaction_status = 'SUCCESS' THEN 1 END) * 100.0 / COUNT(*) < 95 THEN 'SUCCESS_RATE_LOW'
        WHEN COUNT(CASE WHEN fraud_flag = 1 THEN 1 END) * 100.0 / COUNT(*) > 1 THEN 'FRAUD_RATE_HIGH'
        WHEN COUNT(*) < 1000 THEN 'VOLUME_LOW'
        ELSE 'HEALTHY'
    END as system_status
    
FROM phonepe_transactions
WHERE timestamp >= NOW() - INTERVAL 1 HOUR

UNION ALL

SELECT 
    'LAST_24_HOURS' as time_period,
    COUNT(*) as transaction_count,
    SUM(amount_inr) as total_value,
    ROUND(AVG(amount_inr), 2) as avg_transaction_value,
    ROUND(COUNT(CASE WHEN transaction_status = 'SUCCESS' THEN 1 END) * 100.0 / COUNT(*), 2) as success_rate,
    ROUND(COUNT(CASE WHEN fraud_flag = 1 THEN 1 END) * 100.0 / COUNT(*), 4) as fraud_rate,
    COUNT(DISTINCT sender_bank) as active_banks,
    COUNT(DISTINCT merchant_category) as active_merchant_categories,
    
    CASE 
        WHEN COUNT(CASE WHEN transaction_status = 'SUCCESS' THEN 1 END) * 100.0 / COUNT(*) < 95 THEN 'SUCCESS_RATE_LOW'
        WHEN COUNT(CASE WHEN fraud_flag = 1 THEN 1 END) * 100.0 / COUNT(*) > 1 THEN 'FRAUD_RATE_HIGH'
        WHEN COUNT(*) < 50000 THEN 'VOLUME_LOW'
        ELSE 'HEALTHY'
    END as system_status
    
FROM phonepe_transactions
WHERE timestamp >= NOW() - INTERVAL 24 HOUR;

-- 11. Competitive Intelligence Dashboard
CREATE VIEW competitive_landscape AS
WITH phonepe_performance AS (
    SELECT 
        'PhonePe' as platform,
        COUNT(*) as monthly_transactions,
        SUM(amount_inr) as monthly_value,
        ROUND(AVG(amount_inr), 2) as avg_transaction_value,
        ROUND(COUNT(CASE WHEN transaction_status = 'SUCCESS' THEN 1 END) * 100.0 / COUNT(*), 2) as success_rate
    FROM phonepe_transactions
    WHERE timestamp >= CURDATE() - INTERVAL 30 DAY
),
market_comparison AS (
    SELECT 
        platform,
        monthly_transactions,
        monthly_value,
        avg_transaction_value,
        success_rate,
        CASE 
            WHEN platform = 'PhonePe' THEN 46.46
            WHEN platform = 'Google Pay' THEN 35.56
            WHEN platform = 'Paytm' THEN 6.90
            ELSE 11.08  
        END as market_share_estimate
    FROM phonepe_performance
    
    UNION ALL
    
    -- Estimated competitor metrics (replace with actual data when available)
    SELECT 'Google Pay', monthly_transactions * 0.77, monthly_value * 0.77, avg_transaction_value * 0.95, success_rate * 0.98, 35.56
    FROM phonepe_performance
    
    UNION ALL
    
    SELECT 'Paytm', monthly_transactions * 0.15, monthly_value * 0.15, avg_transaction_value * 0.85, success_rate * 0.96, 6.90
    FROM phonepe_performance
)
SELECT 
    platform,
    monthly_transactions,
    monthly_value,
    avg_transaction_value,
    success_rate,
    market_share_estimate,
    CASE 
        WHEN platform = 'PhonePe' THEN 'LEADER'
        WHEN market_share_estimate > 30 THEN 'MAJOR_COMPETITOR'
        WHEN market_share_estimate > 5 THEN 'SIGNIFICANT_PLAYER'
        ELSE 'MINOR_PLAYER'
    END as competitive_position
FROM market_comparison
ORDER BY market_share_estimate DESC;


-- 12. Query Performance Monitoring
EXPLAIN ANALYZE 
SELECT 
    sender_state,
    merchant_category,
    COUNT(*) as transaction_count,
    SUM(amount_inr) as total_value
FROM phonepe_transactions
WHERE timestamp >= CURDATE() - INTERVAL 7 DAY
    AND fraud_flag = 0
GROUP BY sender_state, merchant_category
ORDER BY total_value DESC
LIMIT 100;

