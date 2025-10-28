# basket_analysis.py
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from config import AUGMENTED_DATA_PATH
import warnings

def run_basket_analysis():
    """
    Thực hiện Phân tích giỏ hàng (2.1) và Gợi ý (2.2).
    """
    warnings.filterwarnings('ignore')
    sns.set_style("whitegrid")
    
    print("  [Basket] 2.1: Chuẩn bị dữ liệu FP-Growth...")
    
    # TỐI ƯU HÓA: Chỉ scan 2 cột cần cho FP-Growth
    try:
        lazy_df_basket = pl.scan_parquet(str(AUGMENTED_DATA_PATH)).select([
            "order_id", "category4"
        ])
    except Exception as e:
        print(f"  [Basket] Lỗi khi đọc file: {e}. Bạn đã chạy 'preprocess.py' chưa?")
        return

    # === BẮT ĐẦU CODE MỤC 2.1 (CHUẨN BỊ) ===
    
    # gom nhóm theo order_id và list sản phẩm
    print("    - Đang gom nhóm giao dịch (có thể mất vài phút)...") 
    transaction_data = (
        lazy_df_basket  # <--- SỬA TỪ df
        .select(["order_id", "category4"])  
        .group_by("order_id")
        .agg(pl.col("category4").alias("items"))
        .collect(streaming=True) # <--- THÊM VÀO
    )

    transactions = transaction_data["items"].to_list()
    transactions = [t for t in transactions if len(t) > 1]
    print(f"    - Số giao dịch sau khi lọc: {len(transactions)}")

    np.random.seed(42)
    SAMPLE_SIZE = 500_000 # 500k mẫu giao dịch
    TOP_N_ITEMS = 500

    actual_sample_size = min(SAMPLE_SIZE, len(transactions))
    if actual_sample_size < SAMPLE_SIZE:
        print(f"    - Số lượng giao dịch không đủ dữ liệu. Lấy mẫu tối đa: {actual_sample_size:,}")

    sample_idx = np.random.choice(len(transactions), size=actual_sample_size, replace=False)
    transactions_sample_raw = [transactions[i] for i in sample_idx]
    print(f"    - Số giao dịch mẫu đã lấy: {len(transactions_sample_raw):,}")

    all_items = [item for t in transactions_sample_raw for item in t]
    item_counts = Counter(all_items)
    top_items = set([item for item, count in item_counts.most_common(TOP_N_ITEMS)])

    filtered_transactions = [
        [item for item in t if item in top_items]
        for t in transactions_sample_raw
    ]
    filtered_transactions = [t for t in filtered_transactions if len(t) > 1] 

    print(f"    - Số giao dịch cuối cùng: {len(filtered_transactions):,}")
    print(f"    - Số lượng item duy nhất tham gia phân tích: {len(top_items)}")
    
    # === KẾT THÚC CODE MỤC 2.1 (CHUẨN BỊ) ===


    print("  [Basket] 2.1: Chạy FP-Growth...")
    
    # === BẮT ĐẦU CODE MỤC 2.1 (FP-GROWTH) ===

    # mã hóa dữ liệu 
    te = TransactionEncoder()
    te_ary = te.fit(filtered_transactions).transform(filtered_transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # dùng thuật toán FP-Growth sinh luật kết hợp
    MIN_SUPPORT = 0.02 
    frequent_itemsets = fpgrowth(df_trans, min_support=MIN_SUPPORT, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)
    print(f"    - Có {len(frequent_itemsets)} tập phổ biến")

    rules_500 = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules_500 = rules_500.sort_values("lift", ascending=False)
    print(f"    - Tìm thấy {len(rules_500)} luật kết hợp")

    print("\nTop 10 luật kết hợp mạnh nhất:")
    print(rules_500.head(10)) # <--- SỬA TỪ display()

    # (Vẽ biểu đồ)
    top10_itemsets = frequent_itemsets.sort_values("support", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        [' + '.join(list(i)) for i in top10_itemsets['itemsets']],
        top10_itemsets['support'] * 100
    )
    plt.gca().invert_yaxis()
    plt.title("Top 10 tập phổ biến (mẫu 500k đơn, top 500 items)", fontsize=14)
    plt.xlabel("Support (%)")
    plt.ylabel("Itemsets")
    
    for bar in bars:
        plt.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}%",
            va='center'
        )
    
    plt.tight_layout()
    plt.show()

    # === KẾT THÚC CODE MỤC 2.1 (FP-GROWTH) ===


    # Giải phóng bộ nhớ trước khi làm phần 2.2
    del lazy_df_basket, transaction_data, transactions, filtered_transactions, df_trans
    
    
    print("  [Basket] 2.2: Chuẩn bị dữ liệu Recommender...")
    # TỐI ƯU HÓA: Chỉ scan 2 cột cần cho Recommender
    try:
        lazy_df_rec = pl.scan_parquet(str(AUGMENTED_DATA_PATH)).select([
            "user_id", "category4"
        ])
    except Exception as e:
        print(f"  [Basket] Lỗi khi đọc file: {e}.")
        return

    # === BẮT ĐẦU CODE MỤC 2.2 (CHUẨN BỊ) ===
    
    # lấy ngẫu nhiên 50K người
    SAMPLE_USERS = 50_000
    print("    - Lấy danh sách user IDs...")
    all_user_ids = lazy_df_rec.select("user_id").unique().collect(streaming=True)["user_id"].to_numpy() # <--- SỬA TỪ df
    
    np.random.seed(42)
    sampled_user_ids = np.random.choice(all_user_ids, size=SAMPLE_USERS, replace=False)

    print("    - Lấy mẫu 50k user...")
    df_sampled = lazy_df_rec.filter(pl.col('user_id').is_in(sampled_user_ids)).collect(streaming=True) # <--- SỬA TỪ df VÀ THÊM collect

    df_pandas = df_sampled.to_pandas()
    df_pandas['user_id'] = df_pandas['user_id'].astype('category')
    df_pandas['category4'] = df_pandas['category4'].astype('category')
    print(f"    - Đã lấy mẫu {len(df_pandas['user_id'].unique())} người dùng với {len(df_pandas)} dòng")

    user_item_interaction = df_pandas.drop_duplicates()
    user_item_interaction['purchased'] = 1 

    # === KẾT THÚC CODE MỤC 2.2 (CHUẨN BỊ) ===

    print("  [Basket] 2.2: Tính toán Cosine Similarity...")
    
    # === BẮT ĐẦU CODE MỤC 2.2 (TÍNH TOÁN) ===

    # ma trận user-item
    user_item_matrix = user_item_interaction.pivot_table(
        index='user_id',
        columns='category4',
        values='purchased',
        fill_value=0
    )

    user_mean = user_item_matrix.mean(axis=1)
    user_item_matrix_adjusted = user_item_matrix.subtract(user_mean, axis=0)
    print("    - 5 dòng đầu ma trận chuẩn hóa:")
    print(user_item_matrix_adjusted.head()) # <--- SỬA TỪ display()

    # ma trận item-user
    item_user_matrix = user_item_matrix_adjusted.T
    item_user_sparse_matrix = csr_matrix(item_user_matrix.values)

    # tính mức độ tương đồng item-item bằng cosine
    item_similarity = cosine_similarity(item_user_sparse_matrix)

    item_similarity_df = pd.DataFrame(
        item_similarity, 
        index=item_user_matrix.index,
        columns=item_user_matrix.index
    )
    
    print("    - 5 dòng đầu ma trận tương đồng:")
    print(item_similarity_df.head()) # <--- SỬA TỪ display()
    
    # === KẾT THÚC CODE MỤC 2.2 (TÍNH TOÁN) ===

    # === BẮT ĐẦU CODE MỤC 2.2 (HÀM GỢI Ý) ===
    
    def recommend_items(item_name, similarity_df, num_recommendations=10):
        if item_name not in similarity_df.columns:
            return f"Sản phẩm '{item_name}' không có trong dữ liệu mẫu"
        
        similar_scores = similarity_df[item_name]
        similar_scores = similar_scores.sort_values(ascending=False)
        recommendations = similar_scores[1:num_recommendations+1]
        
        return recommendations.to_frame(name='Điểm tương đồng')

    recommendations = recommend_items('MACUNLAR', item_similarity_df)

    print("\nNếu người dùng mua 'MACUNLAR', gợi ý các sản phẩm sau:")
    print(recommendations) # <--- SỬA TỪ display()

    # === KẾT THÚC CODE MỤC 2.2 (HÀM GỢI Ý) ===

    print("  [Basket] Hoàn tất phân tích giỏ hàng.")