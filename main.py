# main.py
import warnings
import time

# Nhập các module bạn vừa tạo
import preprocess
import revenue_analysis
import revenue_prediction
import basket_analysis
import customer_clustering

def main():
    warnings.filterwarnings('ignore')
    start_time_total = time.time()
    
    # === BƯỚC 0: TIỀN XỬ LÝ (CHẠY 1 LẦN) ===
    # Tính toán và lưu file parquet mới với cột revenue
    # Nếu file đã tồn tại, hàm này sẽ tự động bỏ qua
    print("Bắt đầu Bước 0: Tiền xử lý (Tính toán doanh thu)...")
    start_time = time.time()
    preprocess.calculate_and_save_revenue()
    print(f"Hoàn tất Bước 0. Thời gian: {time.time() - start_time:.2f} giây")

    # === BƯỚC 1: PHÂN TÍCH DOANH THU ===
    print("\nBắt đầu Bước 1: Phân tích doanh thu (Mục 1.2 - 1.6)...")
    start_time = time.time()
    revenue_analysis.run_revenue_analysis()
    print(f"Hoàn tất Bước 1. Thời gian: {time.time() - start_time:.2f} giây")
    # -> RAM được giải phóng tại đây

    # === BƯỚC 2: DỰ ĐOÁN DOANH THU ===
    print("\nBắt đầu Bước 2: Dự đoán doanh thu (Mục 1.7 - 1.8)...")
    start_time = time.time()
    revenue_prediction.run_revenue_prediction()
    print(f"Hoàn tất Bước 2. Thời gian: {time.time() - start_time:.2f} giây")
    # -> RAM được giải phóng tại đây

    # === BƯỚC 3: PHÂN TÍCH GIỎ HÀNG ===
    print("\nBắt đầu Bước 3: Phân tích giỏ hàng (Mục 2)...")
    start_time = time.time()
    basket_analysis.run_basket_analysis()
    print(f"Hoàn tất Bước 3. Thời gian: {time.time() - start_time:.2f} giây")
    # -> RAM được giải phóng tại đây

    # === BƯỚC 4: PHÂN CỤM KHÁCH HÀNG ===
    print("\nBắt đầu Bước 4: Phân cụm khách hàng (Mục 3)...")
    start_time = time.time()
    customer_clustering.run_customer_clustering()
    print(f"Hoàn tất Bước 4. Thời gian: {time.time() - start_time:.2f} giây")
    # -> RAM được giải phóng tại đây

    print(f"\n*** TOÀN BỘ QUÁ TRÌNH HOÀN TẤT! ***")
    print(f"Tổng thời gian chạy: {time.time() - start_time_total:.2f} giây")

if __name__ == "__main__":
    main()