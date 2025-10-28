# customer_clustering.py
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
from config import AUGMENTED_DATA_PATH
import warnings

def run_customer_clustering():
    """
    Thực hiện Phân cụm khách hàng (RFM & K-Means - Mục 3).
    """
    warnings.filterwarnings('ignore')
    sns.set_style("whitegrid")
    
    print("  [Clustering] Đang tải dữ liệu (scan)...")
    
    try:
        lazy_df = pl.scan_parquet(str(AUGMENTED_DATA_PATH)).select([
            "user_id", "order_date", "order_id", "net_revenue"
        ])
    except Exception as e:
        print(f"  [Clustering] Lỗi khi đọc file: {e}. Bạn đã chạy 'preprocess.py' chưa?")
        return
        
    # === 3.1. Phân tích RFM ===
    print("  [Clustering] 3.1: Tính toán RFM...")
    try:
        # 1. Tính toán ngày snapshot
        snapshot_date = lazy_df.select(pl.max("order_date")).collect(streaming=True).item() + timedelta(days=1)
        print(f"    - Snapshot date (Last order date + 1): {snapshot_date}")

        # 2. Tính toán R, F, M
        print("    - Đang tính toán R, F, M...")
        rfm_df = lazy_df.group_by("user_id").agg([
            (snapshot_date - pl.max("order_date")).dt.total_days().alias("Recency"),
            pl.col("order_id").n_unique().alias("Frequency"),
            pl.col("net_revenue").sum().alias("Monetary")
        ]).collect(streaming=True) 

        print(f"    - Tính xong RFM cho {rfm_df.height} khách hàng.")
        
        # === PRINT HEAD (1) - ĐÃ THÊM ===
        print("    - 5 dòng đầu của dữ liệu RFM (rfm_df):")
        print(rfm_df.head())

        # 3. Tính điểm R, F, M
        print("    - Đang tính điểm RFM (1-5)...")
        rfm_pd = rfm_df.to_pandas()
        
        rfm_pd['R_score'] = pd.qcut(rfm_pd['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
        rfm_pd['F_score'] = pd.qcut(rfm_pd['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm_pd['M_score'] = pd.qcut(rfm_pd['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)

        # 4. Tạo điểm tổng hợp
        rfm_df = pl.from_pandas(rfm_pd) 
        rfm_df = rfm_df.with_columns(
            RFM_Score = pl.col("R_score").cast(str) + pl.col("F_score").cast(str) + pl.col("M_score").cast(str),
            RFM_Sum = pl.col("R_score") + pl.col("F_score") + pl.col("M_score")
        )
        print("    - Đã tính xong điểm RFM.")
        
        # === PRINT HEAD (2) - ĐÃ THÊM ===
        print("    - 5 dòng đầu của rfm_df sau khi tính điểm:")
        print(rfm_df.head())

        # 5. Định nghĩa các phân khúc
        rfm_pd = rfm_df.to_pandas()
        
        seg_map_simple = {
            r'[45][45]': 'Khách VIP (Champions)',
            r'[345][345]': 'Khách hàng trung thành & Tiềm năng (Loyal & Potential)',
            r'[45][12]': 'Khách hàng mới & Triển vọng (New & Promising)', 
            r'[123][345]': 'Khách hàng có nguy cơ (At Risk)',
            r'[12][12]': 'Khách hàng đã mất (Lost)'
        }

        rfm_pd['Segment'] = rfm_pd['R_score'].astype(str) + rfm_pd['F_score'].astype(str)
        rfm_pd['Segment'] = rfm_pd['Segment'].replace(seg_map_simple, regex=True)
        rfm_pd.loc[rfm_pd['Segment'].str.match(r'^[1-5]{2}$'), 'Segment'] = 'Khách hàng khác (Others)'
        
        rfm_df = pl.from_pandas(rfm_pd)

        # 6. Phân tích các phân khúc
        print("\nPhân tích đặc điểm các phân cụm RFM:")
        segment_analysis = rfm_df.group_by("Segment").agg([
            pl.mean("Recency").alias("Avg_Recency"),
            pl.mean("Frequency").alias("Avg_Frequency"),
            pl.mean("Monetary").alias("Avg_Monetary"),
            pl.len().alias("Customer_Count")
        ]).sort("Customer_Count", descending=True)
        
        print(segment_analysis) 

        # 7. Trực quan hóa phân bố phân khúc
        segment_counts_pd = segment_analysis.select(["Segment", "Customer_Count"]).to_pandas()
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=segment_counts_pd, y="Segment", x="Customer_Count", palette="viridis")
        plt.title("Phân Bố Khách Hàng Theo Phân Cụm RFM (3.1)")
        plt.xlabel("Số Lượng Khách Hàng")
        plt.ylabel("Phân Cụm")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        for i, (count, segment) in enumerate(zip(segment_counts_pd['Customer_Count'], segment_counts_pd['Segment'])):
            plt.text(count, i, f' {count:,}', ha='left', va='center')
        
        plt.show() 
        
    except Exception as e:
        print(f"  [Clustering] Lỗi RFM: {e}")

    # === 3.2. Phân cụm K-Means ===
    print("  [Clustering] 3.2: Chuẩn bị dữ liệu K-Means...")
    try:
        
        rfm_for_cluster = rfm_df.select([
            pl.col("Recency"),
            (pl.col("Frequency") + 1).log().alias("Frequency_log"), 
            (pl.col("Monetary") + 1).log().alias("Monetary_log") 
        ]).to_pandas()

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_for_cluster)
        print("    - Dữ liệu RFM đã được Log Transform và Chuẩn hóa (rfm_scaled).")

        # 3. Tìm K tối ưu (Elbow Method)
        print("  [Clustering] 3.2: Chạy Elbow Method...")
        wcss = []
        K_range = range(1, 11) 
        for k in K_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10) 
            kmeans.fit(rfm_scaled)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(K_range, wcss, marker='o', linestyle='--')
        plt.title('Phương pháp Elbow tìm K tối ưu (3.2)')
        plt.xlabel('Số lượng cụm (K)')
        plt.ylabel('WCSS (Inertia)')
        plt.grid(True)
        plt.xticks(K_range)
        plt.show() 
        print("    - Đã vẽ biểu đồ Elbow.")
        
        # 4. Chạy K-Means với K tối ưu 
        OPTIMAL_K = 3 
        
        print(f"  [Clustering] 3.2: Chạy K-Means (K={OPTIMAL_K})...")
        kmeans = KMeans(n_clusters=OPTIMAL_K, init='k-means++', random_state=42, n_init=10)
        clusters = kmeans.fit_predict(rfm_scaled)

        # Thêm cột Cluster vào rfm_df
        rfm_df = rfm_df.with_columns(pl.Series("Cluster_KMeans", clusters))

        # 5. Phân tích đặc điểm các cụm K-Means
        print("\nPhân tích đặc điểm các cụm K-Means:")
        cluster_analysis_kmeans = rfm_df.group_by("Cluster_KMeans").agg([
            pl.mean("Recency").alias("Avg_Recency"),
            pl.mean("Frequency").alias("Avg_Frequency"),
            pl.mean("Monetary").alias("Avg_Monetary"),
            pl.len().alias("Customer_Count")
        ]).sort("Cluster_KMeans")
        
        print(cluster_analysis_kmeans) 

        # 6. Vẽ biểu đồ K-Means
        cluster_analysis_pd = cluster_analysis_kmeans.to_pandas().set_index("Cluster_KMeans")
        cluster_analysis_pd.index = cluster_analysis_pd.index.astype(str)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Phân Tích Đặc Điểm Các Cụm K-Means (3.2)', fontsize=20, y=1.03)

        sns.barplot(data=cluster_analysis_pd, y=cluster_analysis_pd.index, x="Customer_Count", orient='h', ax=axes[0, 0], palette="Spectral").set_title('Số Lượng Khách Hàng')
        sns.barplot(data=cluster_analysis_pd, y=cluster_analysis_pd.index, x="Avg_Recency", orient='h', ax=axes[0, 1], palette="coolwarm_r").set_title('Recency Trung Bình (Càng thấp càng tốt)')
        sns.barplot(data=cluster_analysis_pd, y=cluster_analysis_pd.index, x="Avg_Frequency", orient='h', ax=axes[1, 0], palette="viridis").set_title('Frequency Trung Bình')
        sns.barplot(data=cluster_analysis_pd, y=cluster_analysis_pd.index, x="Avg_Monetary", orient='h', ax=axes[1, 1], palette="plasma").set_title('Monetary Trung Bình')

        for ax in axes.flat:
            ax.set_ylabel('Cụm (Cluster)')

        plt.tight_layout()
        plt.show() 
        
    except Exception as e:
        print(f"Lỗi khi chạy K-Means hoặc phân tích cụm: {e}")

    print("  [Clustering] Hoàn tất phân cụm khách hàng.")