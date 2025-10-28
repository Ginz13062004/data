# preprocess.py
import polars as pl
from config import FINAL_DATA_PATH, AUGMENTED_DATA_PATH

def calculate_and_save_revenue():
    """
    Tải dữ liệu gốc, tính toán doanh thu (gross/net) và lưu ra file Parquet mới.
    Đây là bước tiền xử lý chạy một lần.
    """
    if AUGMENTED_DATA_PATH.exists():
        print(f"File '{AUGMENTED_DATA_PATH.name}' đã tồn tại. Bỏ qua bước tiền xử lý.")
        return

    print(f"Đang đọc file: {FINAL_DATA_PATH} (sử dụng scan)")
    # Sử dụng scan() để thực thi lười biếng (lazy)
    df_lazy = pl.scan_parquet(str(FINAL_DATA_PATH))

    # === Dán code từ mục 1.1 của bạn vào đây ===
    # 1. gross revenue
    df_lazy = df_lazy.with_columns(
        (pl.col('amount') * pl.col('unit_price')).alias('gross_revenue')
    )

    # 2. total gross revenue per order
    df_lazy = df_lazy.with_columns(
        pl.sum("gross_revenue").over("order_id").alias("total_gross_revenue_per_order")
    )

    # 3. net revenue
    df_lazy = df_lazy.with_columns(
        pl.when(pl.col("total_gross_revenue_per_order") > 0)
          .then(
              (pl.col("gross_revenue") / pl.col("total_gross_revenue_per_order")) * pl.col("total_basket")
          )
          .otherwise(0)
          .alias("net_revenue")
    )

    # 4. discount_amount
    df_lazy = df_lazy.with_columns(
        (pl.col("gross_revenue") - pl.col("net_revenue")).alias("discount_amount")
    )

    # 5. Bỏ cột trung gian
    df_lazy = df_lazy.drop('total_gross_revenue_per_order')
    # === Kết thúc code mục 1.1 ===

    # 6. Thu thập kết quả và lưu
    print("Đang thực thi tính toán và lưu file mới (có thể mất vài phút)...")
    # .collect(streaming=True) sẽ xử lý từng khối (chunk) để tránh tràn RAM
    df_augmented = df_lazy.collect(streaming=True) 
    
    print(f"Đang lưu vào: {AUGMENTED_DATA_PATH}")
    df_augmented.write_parquet(str(AUGMENTED_DATA_PATH))
    print("Lưu file tiền xử lý thành công.")

if __name__ == "__main__":
    # Dòng này cho phép bạn chạy riêng file này
    # gõ "python preprocess.py" trong terminal
    calculate_and_save_revenue()