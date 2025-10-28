# config.py
from pathlib import Path
import polars as pl

# Đường dẫn đến thư mục data_clean
DATA_CLEAN = Path(r"D:\datamining-fix\data-mining\source\data_clean")

# File dữ liệu gốc (đã clean)
FINAL_DATA_PATH = DATA_CLEAN / "final_data.parquet"

# File dữ liệu mới (sau khi đã tính thêm doanh thu)
# Module preprocess.py sẽ tạo ra file này
AUGMENTED_DATA_PATH = DATA_CLEAN / "final_data_with_revenue.parquet"

# Tắt một số cảnh báo của Polars
#pl.Config.set_warn_for_potential_instability(False)