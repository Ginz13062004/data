# revenue_prediction.py
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from config import AUGMENTED_DATA_PATH
from datetime import date
from dateutil.relativedelta import relativedelta  
import matplotlib.ticker as mticker
import warnings

def run_revenue_prediction():
    """
    Thực hiện dự đoán doanh thu (Mục 1.7 - 1.8).
    Tối ưu: Tải 2 cột, tổng hợp trực tiếp theo tháng.
    """
    warnings.filterwarnings('ignore')
    sns.set_style("whitegrid")

    print("  [Prediction] Đang tải dữ liệu (scan)...")
    
    try:
        lazy_df = pl.scan_parquet(str(AUGMENTED_DATA_PATH)).select([
            "order_date", "net_revenue"
        ])
    except Exception as e:
        print(f"  [Prediction] Lỗi khi đọc file: {e}. Bạn đã chạy 'preprocess.py' chưa?")
        return

    # === 1.7.3 TỔNG HỢP THEO THÁNG (Đã tối ưu) ===
    print("  [Prediction] Tổng hợp doanh thu hàng tháng...")
    monthly = (
        lazy_df
        .filter(pl.col("order_date").is_not_null())
        .with_columns(pl.col("order_date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.col("net_revenue").sum().alias("net_revenue"))
        .sort("month")
        .collect(streaming=True) # Kết quả chỉ có ~36 dòng, rất nhẹ
    )
    print(f"  [Prediction] Đã tổng hợp {monthly.height} tháng.")
    print(monthly.head())

    # === 1.7.4 CHIA TRAIN / TEST ===
    monthly = monthly.with_columns(
        t = pl.int_range(0, pl.count())  
    )
    test_size = 6 if monthly.height > 12 else max(1, monthly.height // 5)
    train = monthly.slice(0, monthly.height - test_size)
    test = monthly.slice(monthly.height - test_size)

    X_train = train.select("t")
    y_train = train.select("net_revenue")
    X_test = test.select("t")
    y_test = test.select("net_revenue")
    print(f"    - Tổng tháng: {monthly.height}, Train: {train.height}, Test: {test.height}")

    # === 1.7.5 HUẤN LUYỆN MÔ HÌNH ===
    print("  [Prediction] Huấn luyện mô hình Linear Regression...")
    model = LinearRegression()
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    
    intercept = float(model.intercept_)
    coef = float(model.coef_.ravel()[0])
    print(f"    - Phương trình: net_revenue = {intercept:.2f} + {coef:.2f} * t")
    
    # === 1.7.7 ĐÁNH GIÁ MÔ HÌNH ===
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    if y_pred_train.ndim > 1: y_pred_train = y_pred_train.flatten()
    if y_pred_test.ndim > 1: y_pred_test = y_pred_test.flatten()

    df_train_eval = pl.DataFrame({"actual": y_train.to_series().to_numpy().flatten().tolist(), "pred": y_pred_train.tolist()})
    df_test_eval = pl.DataFrame({"actual": y_test.to_series().to_numpy().flatten().tolist(), "pred": y_pred_test.tolist()})

    df_train_eval = df_train_eval.with_columns(((pl.col("actual") - pl.col("pred")) ** 2).alias("sq_error"))
    df_test_eval = df_test_eval.with_columns(((pl.col("actual") - pl.col("pred")) ** 2).alias("sq_error"))

    rmse_train = df_train_eval["sq_error"].mean() ** 0.5
    rmse_test = df_test_eval["sq_error"].mean() ** 0.5
    
    ss_res = ((df_test_eval["actual"] - df_test_eval["pred"]) ** 2).sum()
    ss_tot = ((df_test_eval["actual"] - df_test_eval["actual"].mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    print("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (1.7) ===")
    print(f"    - RMSE (Train): {rmse_train:,.2f}")
    print(f"    - RMSE (Test):  {rmse_test:,.2f}")
    print(f"    - R² (Test):    {r2:.4f}")   

    # === 1.7.8 VẼ BIỂU ĐỒ ===
    plt.figure(figsize=(11,5))
    plt.plot(monthly["month"], monthly["net_revenue"], marker="o", label="Actual")
    plt.plot(train["month"], y_pred_train, linestyle="--", label="Fit (Train)")
    plt.plot(test["month"], y_pred_test, marker="x", linestyle="--", label="Pred (Test)")
    plt.title("1.7 Linear Regression — Actual vs Fit/Pred")
    plt.xlabel("Month")
    plt.ylabel("Net Revenue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() # <--- VẼ BIỂU ĐỒ

    # === 1.8 DỰ ĐOÁN 6 THÁNG TỚI ===
    print("  [Prediction] Dự đoán 6 tháng tới...")
    
    # 1.8.1 TẠO DỮ LIỆU
    num_future_months = 6
    last_t = monthly["t"][-1]
    future_t = np.arange(last_t + 1, last_t + 1 + num_future_months).reshape(-1, 1)
    last_month_date = monthly["month"].to_list()[-1]
    future_months_dates = [last_month_date + relativedelta(months=i) for i in range(1, num_future_months + 1)]

    # 1.8.2 DỰ ĐOÁN
    future_revenue_pred = model.predict(future_t)
    future_predictions_df = pl.DataFrame({
        "month_future": future_months_dates,
        "t_future": future_t.ravel(),
        "predicted_revenue": future_revenue_pred.flatten() # <--- Đảm bảo 1D
    })
    print("=== DỰ ĐOÁN DOANH THU 6 THÁNG TỚI (1.8) ===")
    print(future_predictions_df)

    # 1.8.3 VẼ BIỂU ĐỒ
    monthly_pd = monthly.to_pandas()
    future_predictions_pd = future_predictions_df.to_pandas()
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_pd["month"], monthly_pd["net_revenue"], marker="o", linestyle="-", label="Doanh thu thực tế (Actual)")
    plt.plot(monthly_pd["month"][:-test_size], y_pred_train, linestyle="--", label="Đường hồi quy (Fit - Train)")
    plt.plot(monthly_pd["month"][-test_size:], y_pred_test, linestyle="--", marker="x", label="Dự đoán (Pred - Test)")
    plt.plot(future_predictions_pd["month_future"], future_predictions_pd["predicted_revenue"], linestyle=":", marker="*", color="red", label="Dự đoán 6 tháng tới (Forecast)")

    formatter = mticker.FuncFormatter(lambda x, p: f'{x/1e6:,.0f}M') # Hiển thị theo triệu (M)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.title("1.8 – Dự Đoán Doanh Thu 6 Tháng Tới (Linear Regression)", fontsize=16)
    plt.xlabel("Tháng", fontsize=12)
    plt.ylabel("Doanh thu ròng (Net Revenue)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.show() # <--- VẼ BIỂU ĐỒ
    
    print("  [Prediction] Hoàn tất dự đoán doanh thu.")