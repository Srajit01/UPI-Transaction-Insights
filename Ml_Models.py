# ML Models for Fraud Detection

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import networkx as nx
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PhonePeAdvancedFraudDetection:
    """
    Advanced fraud detection system specifically designed for PhonePe transactions
    Incorporates multiple ML techniques including anomaly detection, clustering, 
    and graph-based network analysis
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.is_trained = False
        
    def prepare_features(self, df):
        """
        Enhanced feature engineering for PhonePe transactions
        """
        features_df = df.copy()
        
        # Time-based features
        features_df['hour'] = pd.to_datetime(features_df['timestamp']).dt.hour
        features_df['day_of_week'] = pd.to_datetime(features_df['timestamp']).dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        features_df['is_night'] = ((features_df['hour'] >= 22) | (features_df['hour'] <= 6)).astype(int)
        
        # Amount-based features
        features_df['amount_log'] = np.log1p(features_df['amount_inr'])
        features_df['amount_zscore'] = (features_df['amount_inr'] - features_df['amount_inr'].mean()) / features_df['amount_inr'].std()
        
        # Categorical encoding
        categorical_columns = ['transaction_type', 'merchant_category', 'sender_age_group', 
                             'sender_state', 'sender_bank', 'receiver_bank', 'device_type', 'network_type']
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                features_df[f'{col}_encoded'] = self.encoders[col].fit_transform(features_df[col])
            else:
                features_df[f'{col}_encoded'] = self.encoders[col].transform(features_df[col])
        
        # Cross-state transaction indicator
        features_df['cross_state'] = (features_df['sender_state'] != features_df.get('receiver_state', features_df['sender_state'])).astype(int)
        
        # High-value transaction indicator
        features_df['high_value'] = (features_df['amount_inr'] > features_df['amount_inr'].quantile(0.95)).astype(int)
        
        return features_df
    
    def create_user_profiles(self, df):
        """
        Create user behavioral profiles for anomaly detection
        """
        user_profiles = df.groupby(['sender_age_group', 'sender_state', 'device_type']).agg({
            'amount_inr': ['mean', 'std', 'count', 'max'],
            'transaction_type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'P2P',
            'merchant_category': 'nunique',
            'fraud_flag': 'sum',
            'timestamp': lambda x: (pd.to_datetime(x).max() - pd.to_datetime(x).min()).days
        }).round(2)
        
        user_profiles.columns = ['avg_amount', 'std_amount', 'txn_count', 'max_amount', 
                                'preferred_type', 'merchant_diversity', 'fraud_count', 'active_days']
        
        # Calculate derived metrics
        user_profiles['fraud_rate'] = user_profiles['fraud_count'] / user_profiles['txn_count']
        user_profiles['daily_avg_txns'] = user_profiles['txn_count'] / user_profiles['active_days']
        user_profiles['risk_score'] = (user_profiles['fraud_rate'] * 0.4 + 
                                     (user_profiles['max_amount'] / user_profiles['avg_amount']).clip(0, 5) * 0.3 +
                                     (user_profiles['daily_avg_txns'] / 10).clip(0, 1) * 0.3)
        
        return user_profiles.reset_index()
    
    def build_transaction_graph(self, df):
        """
        Build transaction network graph for fraud cluster detection
        """
        # Create network graph
        G = nx.DiGraph()
        
        # Add edges for bank-to-bank transactions
        for _, row in df.iterrows():
            sender = f"bank_{row['sender_bank']}"
            receiver = f"bank_{row['receiver_bank']}"
            amount = row['amount_inr']
            fraud = row['fraud_flag']
            
            if G.has_edge(sender, receiver):
                G[sender][receiver]['weight'] += 1
                G[sender][receiver]['total_amount'] += amount
                G[sender][receiver]['fraud_count'] += fraud
            else:
                G.add_edge(sender, receiver, weight=1, total_amount=amount, fraud_count=fraud)
        
        # Calculate fraud rates for edges
        for u, v, d in G.edges(data=True):
            d['fraud_rate'] = d['fraud_count'] / d['weight']
        
        return G
    
    def detect_fraud_clusters(self, G, fraud_threshold=0.05):
        """
        Detect suspicious transaction clusters using graph analysis
        """
        suspicious_clusters = []
        
        # Find edges with high fraud rates
        high_fraud_edges = [(u, v, d) for u, v, d in G.edges(data=True) 
                           if d['fraud_rate'] > fraud_threshold and d['weight'] > 10]
        
        # Cluster analysis using centrality measures
        betweenness = nx.betweenness_centrality(G, weight='weight')
        closeness = nx.closeness_centrality(G, distance='weight')
        
        # Identify suspicious nodes
        for node in G.nodes():
            if betweenness[node] > 0.1 and closeness[node] > 0.1:
                # Check fraud rate for this node
                in_edges_fraud = sum(G[u][node].get('fraud_count', 0) for u in G.predecessors(node))
                in_edges_total = sum(G[u][node].get('weight', 0) for u in G.predecessors(node))
                out_edges_fraud = sum(G[node][v].get('fraud_count', 0) for v in G.successors(node))
                out_edges_total = sum(G[node][v].get('weight', 0) for v in G.successors(node))
                
                total_fraud = in_edges_fraud + out_edges_fraud
                total_txns = in_edges_total + out_edges_total
                
                if total_txns > 0:
                    node_fraud_rate = total_fraud / total_txns
                    if node_fraud_rate > fraud_threshold:
                        suspicious_clusters.append({
                            'node': node,
                            'fraud_rate': node_fraud_rate,
                            'betweenness_centrality': betweenness[node],
                            'closeness_centrality': closeness[node],
                            'total_transactions': total_txns
                        })
        
        return sorted(suspicious_clusters, key=lambda x: x['fraud_rate'], reverse=True)
    
    def train_models(self, df):
        """
        Train multiple fraud detection models
        """
        print("Preparing features...")
        features_df = self.prepare_features(df)
        
        # Select features for modeling
        feature_cols = ['amount_log', 'amount_zscore', 'hour', 'day_of_week', 'is_weekend', 'is_night',
                       'transaction_type_encoded', 'merchant_category_encoded', 'sender_age_group_encoded',
                       'sender_state_encoded', 'device_type_encoded', 'network_type_encoded',
                       'cross_state', 'high_value']
        
        X = features_df[feature_cols]
        y = features_df['fraud_flag']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        self.scalers['fraud'] = StandardScaler()
        X_scaled = self.scalers['fraud'].fit_transform(X)
        
        print("Training models...")
        
        # 1. Isolation Forest for anomaly detection
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.002,  # Expected fraud rate
            random_state=42,
            n_estimators=200
        )
        self.models['isolation_forest'].fit(X_scaled)
        
        # 2. Random Forest for supervised learning
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            max_depth=10
        )
        self.models['random_forest'].fit(X_train, y_train)
        
        # Model evaluation
        rf_pred = self.models['random_forest'].predict(X_test)
        rf_scores = self.models['random_forest'].predict_proba(X_test)[:, 1]
        
        print("Random Forest Performance:")
        print(classification_report(y_test, rf_pred))
        print(f"AUC-ROC Score: {roc_auc_score(y_test, rf_scores):.4f}")
        
        # 3. DBSCAN for clustering-based anomaly detection
        self.models['dbscan'] = DBSCAN(eps=0.5, min_samples=5)
        clusters = self.models['dbscan'].fit_predict(X_scaled)
        
        # Identify anomaly clusters
        cluster_fraud_rates = {}
        for cluster_id in np.unique(clusters):
            if cluster_id != -1:  # Not noise
                mask = clusters == cluster_id
                fraud_rate = y[mask].mean()
                cluster_fraud_rates[cluster_id] = fraud_rate
        
        self.anomaly_clusters = [cid for cid, rate in cluster_fraud_rates.items() if rate > 0.05]
        
        # 4. Graph-based fraud detection
        print("Building transaction graph...")
        self.transaction_graph = self.build_transaction_graph(df)
        self.fraud_clusters = self.detect_fraud_clusters(self.transaction_graph)
        
        print(f"Detected {len(self.fraud_clusters)} suspicious transaction clusters")
        
        self.is_trained = True
        return True
    
    def predict_fraud_ensemble(self, df):
        """
        Ensemble prediction combining multiple models
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        features_df = self.prepare_features(df)
        
        feature_cols = ['amount_log', 'amount_zscore', 'hour', 'day_of_week', 'is_weekend', 'is_night',
                       'transaction_type_encoded', 'merchant_category_encoded', 'sender_age_group_encoded',
                       'sender_state_encoded', 'device_type_encoded', 'network_type_encoded',
                       'cross_state', 'high_value']
        
        X = features_df[feature_cols].fillna(features_df[feature_cols].median())
        X_scaled = self.scalers['fraud'].transform(X)
        
        # Get predictions from different models
        iso_scores = self.models['isolation_forest'].decision_function(X_scaled)
        iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
        iso_predictions = 1 - iso_scores_norm  # Higher score = higher fraud risk
        
        rf_predictions = self.models['random_forest'].predict_proba(X_scaled)[:, 1]
        
        # DBSCAN predictions (based on cluster membership)
        clusters = self.models['dbscan'].fit_predict(X_scaled)
        dbscan_predictions = np.array([1.0 if c in self.anomaly_clusters else 0.1 for c in clusters])
        
        # Ensemble prediction (weighted average)
        ensemble_scores = (
            iso_predictions * 0.3 +
            rf_predictions * 0.5 +
            dbscan_predictions * 0.2
        )
        
        # Add graph-based risk scores
        graph_risk_scores = self.calculate_graph_risk_scores(df)
        final_scores = ensemble_scores * 0.8 + graph_risk_scores * 0.2
        
        # Risk categorization
        risk_categories = pd.cut(
            final_scores,
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return final_scores, risk_categories
    
    def calculate_graph_risk_scores(self, df):
        """
        Calculate risk scores based on transaction graph analysis
        """
        risk_scores = []
        
        for _, row in df.iterrows():
            sender_bank = f"bank_{row['sender_bank']}"
            receiver_bank = f"bank_{row['receiver_bank']}"
            
            risk_score = 0.1  # Base risk
            
            # Check if this bank-to-bank connection is in fraud clusters
            for cluster in self.fraud_clusters:
                if cluster['node'] in [sender_bank, receiver_bank]:
                    risk_score += cluster['fraud_rate']
            
            # Check edge fraud rate
            if self.transaction_graph.has_edge(sender_bank, receiver_bank):
                edge_data = self.transaction_graph[sender_bank][receiver_bank]
                edge_fraud_rate = edge_data.get('fraud_rate', 0)
                risk_score += edge_fraud_rate
            
            risk_scores.append(min(risk_score, 1.0))  # Cap at 1.0
        
        return np.array(risk_scores)
    
    def generate_fraud_insights(self, df):
        """
        Generate comprehensive fraud insights and recommendations
        """
        insights = {}
        
        # Overall fraud statistics
        total_transactions = len(df)
        total_fraud = df['fraud_flag'].sum()
        fraud_rate = total_fraud / total_transactions * 100
        
        insights['overview'] = {
            'total_transactions': total_transactions,
            'total_fraud_cases': total_fraud,
            'fraud_rate_percent': round(fraud_rate, 4)
        }
        
        # High-risk patterns
        high_risk_patterns = []
        
        # Time-based patterns
        hourly_fraud = df.groupby(df['timestamp'].dt.hour)['fraud_flag'].agg(['sum', 'count', 'mean'])
        risky_hours = hourly_fraud[hourly_fraud['mean'] > fraud_rate/100].index.tolist()
        if risky_hours:
            high_risk_patterns.append(f"High fraud risk during hours: {risky_hours}")
        
        # Amount-based patterns
        amount_percentiles = df['amount_inr'].quantile([0.5, 0.95, 0.99])
        high_amount_fraud = df[df['amount_inr'] > amount_percentiles[0.95]]['fraud_flag'].mean()
        if high_amount_fraud > fraud_rate/100:
            high_risk_patterns.append(f"High-value transactions (>₹{amount_percentiles[0.95]:.0f}) show elevated fraud risk")
        
        # Geographic patterns
        state_fraud = df.groupby('sender_state')['fraud_flag'].agg(['sum', 'count', 'mean']).sort_values('mean', ascending=False)
        risky_states = state_fraud[state_fraud['mean'] > fraud_rate/100 * 2].head(3).index.tolist()
        if risky_states:
            high_risk_patterns.append(f"States with elevated fraud risk: {', '.join(risky_states)}")
        
        insights['high_risk_patterns'] = high_risk_patterns
        
        # Model performance insights
        if self.is_trained:
            insights['model_insights'] = {
                'fraud_clusters_detected': len(self.fraud_clusters),
                'anomaly_clusters': len(self.anomaly_clusters) if hasattr(self, 'anomaly_clusters') else 0,
                'graph_nodes': self.transaction_graph.number_of_nodes(),
                'graph_edges': self.transaction_graph.number_of_edges()
            }
        
        # Recommendations
        recommendations = [
            "Implement real-time monitoring for high-risk time periods",
            "Enhanced verification for high-value transactions",
            "Geographic risk-based authentication",
            "Graph-based network monitoring for coordinated attacks"
        ]
        
        insights['recommendations'] = recommendations
        
        return insights

# Example usage and testing
def test_phonepe_fraud_detection():
    """
    Test the PhonePe fraud detection system
    """
    # Generate sample data (in production, load from database)
    np.random.seed(42)
    n_samples = 50000
    
    sample_data = {
        'transaction_id': [f'TXN{i:08d}' for i in range(1, n_samples + 1)],
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'transaction_type': np.random.choice(['P2P', 'P2M'], n_samples, p=[0.6, 0.4]),
        'merchant_category': np.random.choice([
            'Grocery', 'Shopping', 'Entertainment', 'Fuel', 'Food', 'Healthcare'
        ], n_samples),
        'amount_inr': np.random.lognormal(6, 1.5, n_samples).astype(int),
        'sender_age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n_samples),
        'sender_state': np.random.choice([
            'Maharashtra', 'Karnataka', 'Delhi', 'Uttar Pradesh', 'Tamil Nadu'
        ], n_samples),
        'sender_bank': np.random.choice(['SBI', 'HDFC', 'ICICI', 'Axis', 'PNB'], n_samples),
        'receiver_bank': np.random.choice(['SBI', 'HDFC', 'ICICI', 'Axis', 'PNB'], n_samples),
        'device_type': np.random.choice(['Android', 'iOS'], n_samples, p=[0.75, 0.25]),
        'network_type': np.random.choice(['4G', '5G', 'WiFi'], n_samples),
        'fraud_flag': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize and train the fraud detection system
    fraud_detector = PhonePeAdvancedFraudDetection()
    
    print("Training PhonePe Advanced Fraud Detection System...")
    fraud_detector.train_models(df)
    
    # Generate predictions on sample data
    print("\nGenerating fraud predictions...")
    fraud_scores, risk_categories = fraud_detector.predict_fraud_ensemble(df.head(1000))
    
    # Generate insights
    print("\nGenerating fraud insights...")
    insights = fraud_detector.generate_fraud_insights(df)
    
    print("\n=== FRAUD DETECTION INSIGHTS ===")
    print(f"Total Transactions: {insights['overview']['total_transactions']:,}")
    print(f"Fraud Cases: {insights['overview']['total_fraud_cases']}")
    print(f"Fraud Rate: {insights['overview']['fraud_rate_percent']:.4f}%")
    
    print("\n=== HIGH-RISK PATTERNS ===")
    for pattern in insights['high_risk_patterns']:
        print(f"• {pattern}")
    
    print("\n=== MODEL INSIGHTS ===")
    for key, value in insights['model_insights'].items():
        print(f"• {key.replace('_', ' ').title()}: {value}")
    
    print("\n=== RECOMMENDATIONS ===")
    for rec in insights['recommendations']:
        print(f"• {rec}")
    
    # Sample predictions
    high_risk_samples = pd.DataFrame({
        'Transaction ID': df.head(1000)['transaction_id'],
        'Amount': df.head(1000)['amount_inr'],
        'Fraud Score': fraud_scores.round(4),
        'Risk Category': risk_categories
    }).sort_values('Fraud Score', ascending=False).head(10)
    
    print("\n=== TOP 10 HIGH-RISK TRANSACTIONS ===")
    print(high_risk_samples.to_string(index=False))

if __name__ == "__main__":
    test_phonepe_fraud_detection()
