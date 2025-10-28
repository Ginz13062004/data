# revenue_analysis.py
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from config import AUGMENTED_DATA_PATH
import warnings

def run_revenue_analysis():
    """
    Thực hiện phân tích doanh thu (Mục 1.2 - 1.6).
    Tải dữ liệu, tổng hợp, vẽ biểu đồ, và giải phóng bộ nhớ.
    """
    warnings.filterwarnings('ignore')
    sns.set_style("whitegrid")
    
    print("  [Revenue] Đang tải dữ liệu (scan)...")
    
    try:
        lazy_df = pl.scan_parquet(str(AUGMENTED_DATA_PATH)).select([
            "order_year", "order_month", "gross_revenue", "net_revenue", 
            "discount_amount", "category1", "order_date", "branch_region"
        ])
    except Exception as e:
        print(f"  [Revenue] Lỗi khi đọc file: {e}. Bạn đã chạy 'preprocess.py' chưa?")
        return

    # === 1.2. Lượng doanh thu chi trả cho chiết khấu hàng tháng ===
    print("  [Revenue] Phân tích 1.2: Chiết khấu hàng tháng...")
    monthly_analysis = (
        lazy_df.group_by(["order_year", "order_month"])
        .agg([
            pl.sum("gross_revenue").alias("total_gross_revenue"),
            pl.sum("net_revenue").alias("total_net_revenue"),
            pl.sum("discount_amount").alias("total_discount")
        ])
        .with_columns(
            pl.when(pl.col("total_gross_revenue") > 0)
            .then(((pl.col("total_discount") / pl.col("total_gross_revenue")) * 100))
            .otherwise(0)
            .alias("overall_discount_rate")
        )
        .sort(["order_year", "order_month"])
        .collect(streaming=True) # <--- THU THẬP DỮ LIỆU
        .to_pandas()
    )
    print("Kết quả phân tích chiết khấu theo tháng:")
    print(monthly_analysis.head(10))

    plt.rcParams['figure.figsize'] = (14, 7)
    ax = sns.barplot(
        data=monthly_analysis,
        x="order_month",
        y="total_discount",
        hue="order_year",
        palette="viridis"
    )
    plt.title("Tổng Lượng Tiền Chiết Khấu Theo Tháng (1.2)", fontsize=16)
    plt.xlabel("Tháng", fontsize=12)
    plt.ylabel("Tổng Tiền Chiết Khấu", fontsize=12)
    plt.xticks(range(0, 12), labels=range(1, 13))
    plt.legend(title="Năm")
    plt.tight_layout()
    plt.show() # <--- VẼ BIỂU ĐỒ

    # === 1.3. Ngành hàng dùng nhiều chiết khấu ===
    print("  [Revenue] Phân tích 1.3: Chiết khấu ngành hàng...")
    category_analysis = (
        lazy_df.group_by("category1")
        .agg([
            pl.sum("gross_revenue").alias("total_gross_revenue"),
            pl.sum("net_revenue").alias("total_net_revenue"),
            pl.sum("discount_amount").alias("total_discount")
        ])
        .with_columns(
            pl.when(pl.col("total_gross_revenue") > 0)
            .then(((pl.col("total_discount") / pl.col("total_gross_revenue")) * 100))
            .otherwise(0)
            .alias("overall_discount_rate")
        )
        .sort("total_discount", descending=True)
        .collect(streaming=True) # <--- THU THẬP DỮ LIỆU
        .to_pandas()
    )
    print("Top 10 ngành hàng 'đốt' nhiều tiền chiết khấu nhất:")
    print(category_analysis.head(10))

    top_10_discount_categories = category_analysis.head(10)
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=top_10_discount_categories,
        x="total_discount",
        y="category1",
        palette="inferno"
    )
    plt.title("Top 10 Ngành Hàng 'Đốt' Nhiều Tiền Chiết Khấu Nhất (1.3)", fontsize=16)
    plt.xlabel("Tổng Tiền Chiết Khấu", fontsize=12)
    plt.ylabel("Ngành Hàng (Category1)", fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show() # <--- VẼ BIỂU ĐỒ
    
    # === 1.4. Doanh thu theo thời gian ===
    print("  [Revenue] Phân tích 1.4: Doanh thu theo thời gian...")
    daily_revenue = (
        lazy_df.group_by(pl.col("order_date").dt.date().alias("order_day"))
        .agg(pl.col("net_revenue").sum().alias("daily_revenue"))
        .sort("order_day")
        .collect(streaming=True) # <--- THU THẬP DỮ LIỆU
        .to_pandas()
    )
    monthly_revenue = (
        lazy_df.group_by(["order_year", "order_month"])
        .agg(pl.col("net_revenue").sum().alias("monthly_revenue"))
        .sort(["order_year", "order_month"])
        .collect(streaming=True) # <--- THU THẬP DỮ LIỆU
        .to_pandas()
    )
    yearly_revenue = (
        lazy_df.group_by("order_year")
        .agg(pl.col("net_revenue").sum().alias("yearly_revenue"))
        .sort("order_year")
        .collect(streaming=True) # <--- THU THẬP DỮ LIỆU
        .to_pandas()
    )
    print("Doanh thu hàng năm")
    print(yearly_revenue)
    print("\nDoanh thu thuần hàng tháng")
    print(monthly_revenue.head())

    plt.figure(figsize=(15, 6))
    plt.plot(daily_revenue["order_day"], daily_revenue["daily_revenue"], label="Doanh thu hàng ngày")
    plt.title("Biến động Doanh thu hàng ngày (1.4)", fontsize=16)
    plt.xlabel("Ngày", fontsize=12)
    plt.ylabel("Doanh thu", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show() # <--- VẼ BIỂU ĐỒ

    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=monthly_revenue,
        x="order_month",
        y="monthly_revenue",
        hue="order_year",
        marker="o",
        palette="magma"
    )
    plt.title("Doanh thu hàng tháng qua các năm (1.4)", fontsize=16)
    plt.xlabel("Tháng", fontsize=12)
    plt.ylabel("Tổng Doanh thu", fontsize=12)
    plt.xticks(range(1, 13))
    plt.legend(title="Năm")
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show() # <--- VẼ BIỂU ĐỒ

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=yearly_revenue,
        x="order_year",
        y="yearly_revenue",
        palette="plasma"
    )
    plt.title("Doanh thu hàng năm (1.4)", fontsize=16)
    plt.xlabel("Năm", fontsize=12)
    plt.ylabel("Tổng Doanh thu", fontsize=12)
    plt.grid(axis='y', alpha=0.5, linestyle='--')
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, 
                bar.get_height(), 
                f'{bar.get_height():,.0f}', 
                ha='center', va='bottom', fontsize=11)
    plt.show() # <--- VẼ BIỂU ĐỒ

    # === 1.5. Doanh thu theo khu vực ===
    print("  [Revenue] Phân tích 1.5: Doanh thu theo khu vực...")
    region_revenue = (
        lazy_df.group_by("branch_region")
        .agg(pl.col("net_revenue").sum().alias("region_revenue"))
        .sort("region_revenue", descending=True)
        .collect(streaming=True) # <--- THU THẬP DỮ LIỆU
        .to_pandas()
    )
    print("Doanh thu theo Vùng")
    print(region_revenue)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=region_revenue,
        x="region_revenue",
        y="branch_region",
        palette="viridis"
    )
    plt.title("Doanh thu theo Vùng (1.5)", fontsize=16)
    plt.xlabel("Tổng Doanh thu", fontsize=12)
    plt.ylabel("Vùng", fontsize=12)
    plt.grid(axis='x', alpha=0.5, linestyle='--')
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 1e7,  
                p.get_y() + p.get_height() / 2,
                f'{width:,.0f}',
                va='center')
    plt.tight_layout()
    plt.show() # <--- VẼ BIỂU ĐỒ

    # === 1.6. Ngành hàng mang lại doanh thu cao nhất ===
    print("  [Revenue] Phân tích 1.6: Ngành hàng doanh thu cao nhất...")
    category_revenue = (
        lazy_df.group_by("category1")
        .agg(pl.col("net_revenue").sum().alias("category_revenue"))
        .sort("category_revenue", descending=True)
        .collect(streaming=True) # <--- THU THẬP DỮ LIỆU
        .to_pandas()
    )
    print("Top 10 Ngành hàng có Doanh thu cao nhất")
    print(category_revenue.head(15))
    
    top_10_categories = category_revenue.head(10)
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(
        data=top_10_categories,
        x="category_revenue",
        y="category1",
        palette="plasma"
    )
    plt.title("Top 10 Ngành Hàng có Doanh thu cao nhất (1.6)", fontsize=16)
    plt.xlabel("Tổng Doanh thu", fontsize=12)
    plt.ylabel("Ngành Hàng (Category1)", fontsize=12)
    plt.grid(axis='x', alpha=0.5, linestyle='--')
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 1e7, 
                p.get_y() + p.get_height() / 2,
                f'{(width / 1e9):.2f}B',
                va='center')
    plt.tight_layout()
    plt.show() # <--- VẼ BIỂU ĐỒ

    print("  [Revenue] Hoàn tất phân tích doanh thu.")