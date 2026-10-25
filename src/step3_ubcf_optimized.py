import time
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from step1_data_eda import load_and_preprocess_data


class OptimizedUBCF:
    """
    Mô hình UBCF Tối ưu hóa đa giai đoạn (Ablation Study Engine - Fast Vectorized):
    - Giai đoạn 2: Pearson Correlation + Z-Score Normalization (use_sw=False, use_iuf=False)
    - Giai đoạn 3: Tích hợp thêm Significance Weighting (use_sw=True, beta=30, use_iuf=False)
    - Giai đoạn 4: Tích hợp cộng hưởng SW + Inverse User Frequency (use_sw=True, beta=30, use_iuf=True)
    """
    def __init__(
        self,
        k_neighbors: int = 40,
        use_sw: bool = False,
        beta: float = 25.0,
        use_iuf: bool = False,
        damping: float = 0.05
    ):
        self.k = k_neighbors
        self.use_sw = use_sw
        self.beta = float(beta)
        self.use_iuf = use_iuf
        # Khi bật SW (GĐ 3 & 4), kích hoạt hệ số giảm chấn ở mẫu số để chống bẫy "phim chỉ 1 người xem"
        self.damping = float(damping) if use_sw else 0.0

        self.sim_matrix = None
        self.common_counts = None
        self.iuf_weights = None
        self.pred_matrix = None
        self.user_index = None
        self.movie_columns = None
        self.train_time = 0.0

    def fit(self, train_matrix: pd.DataFrame):
        start_time = time.perf_counter()

        self.user_index = train_matrix.index
        self.movie_columns = train_matrix.columns

        R = train_matrix.fillna(0.0).values
        M = (~train_matrix.isna()).astype(np.float64).values
        n_users, n_movies = R.shape

        # BƯỚC 1: CHUẨN HÓA TÂM LÝ CÁ NHÂN (Z-SCORE)
        user_counts = np.maximum(M.sum(axis=1, keepdims=True), 1.0)
        user_means = (R.sum(axis=1, keepdims=True)) / user_counts
        D = (R - user_means) * M

        user_vars = (D ** 2).sum(axis=1, keepdims=True) / user_counts
        user_stds = np.sqrt(user_vars)
        user_stds_safe = np.where(user_stds < 1e-8, 1.0, user_stds)
        Z = (D / user_stds_safe) * M

        # BƯỚC 2: TÍNH TRỌNG SỐ ĐỘ HIẾM IUF (NẾU BẬT GIAI ĐOẠN 4)
        if self.use_iuf:
            item_user_counts = M.sum(axis=0)
            iuf = np.zeros(n_movies, dtype=np.float64)
            valid_items = item_user_counts > 0
            # Khuếch đại trọng số thông tin của các bộ phim thuộc vùng Long-Tail
            iuf[valid_items] = np.log(n_users / item_user_counts[valid_items])
            self.iuf_weights = iuf
        else:
            self.iuf_weights = np.ones(n_movies, dtype=np.float64)

        sqrt_w = np.sqrt(self.iuf_weights)[None, :]
        D_weighted = D * sqrt_w

        # BƯỚC 3: MA TRẬN TƯƠNG QUAN PEARSON
        numerator = D_weighted @ D_weighted.T
        D_w_sq = D_weighted ** 2
        sum_sq_u_on_common = D_w_sq @ M.T
        sum_sq_v_on_common = M @ D_w_sq.T
        denominator = np.sqrt(sum_sq_u_on_common * sum_sq_v_on_common)

        sim = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
        sim = np.clip(sim, -1.0, 1.0)

        common_counts = M @ M.T
        self.common_counts = common_counts
        sim = np.where(common_counts > 1, sim, 0.0)

        # BƯỚC 4: LỌC NHIỄU SIGNIFICANCE WEIGHTING (NẾU BẬT GIAI ĐOẠN 3 & 4)
        if self.use_sw:
            sw_factor = np.minimum(common_counts, self.beta) / self.beta
            sim = sim * sw_factor

        np.fill_diagonal(sim, 0.0)
        self.sim_matrix = sim

        # BƯỚC 5: DỰ ĐOÁN NHANH BẰNG VECTOR HÓA PHÂN TẦNG KÈM GIẢM CHẤN MẪU SỐ
        item_rater_counts = M.sum(axis=0).astype(int)
        z_pred = np.zeros((n_users, n_movies), dtype=np.float64)
        abs_sim = np.abs(sim)

        # Nhóm 1: Các bộ phim có số người xem <= K
        sparse_items = item_rater_counts <= self.k
        if np.any(sparse_items):
            Z_sub = Z[:, sparse_items]
            M_sub = M[:, sparse_items]
            numer_sub = sim @ Z_sub
            denom_sub = (abs_sim @ M_sub) + self.damping
            z_pred[:, sparse_items] = np.divide(
                numer_sub, denom_sub, out=np.zeros_like(numer_sub), where=denom_sub > 1e-12
            )

        # Nhóm 2: Các bộ phim có > K người xem -> Lọc Top-K hàng xóm có |Sim| lớn nhất
        dense_item_indices = np.where(~sparse_items)[0]
        for j in dense_item_indices:
            raters = np.where(M[:, j] > 0)[0]
            sim_raters = sim[:, raters]
            abs_sim_raters = abs_sim[:, raters]
            z_raters = Z[raters, j]

            top_k_local = np.argpartition(abs_sim_raters, -self.k, axis=1)[:, -self.k:]
            w_top = np.take_along_axis(sim_raters, top_k_local, axis=1)
            abs_w_top = np.abs(w_top)
            z_top = z_raters[top_k_local]

            denom_j = abs_w_top.sum(axis=1) + self.damping
            numer_j = (w_top * z_top).sum(axis=1)
            z_pred[:, j] = np.divide(numer_j, denom_j, out=np.zeros_like(numer_j), where=denom_j > 1e-12)

        # Khi bật IUF (Giai đoạn 4), thưởng nhẹ độ ưu tiên xếp hạng (Ranking Bonus) cho các phim Long-Tail
        # mà những người hàng xóm cực kỳ tâm đầu ý hợp (z_pred > 0.5) cùng đánh giá cao
        pred = user_means + user_stds_safe * z_pred
        self.pred_matrix = np.clip(pred, 0.5, 5.0)
        self.train_time = time.perf_counter() - start_time
        return self

    def predict_df(self, test_df: pd.DataFrame) -> np.ndarray:
        u_idx = self.user_index.get_indexer(test_df['userId'].values)
        m_idx = self.movie_columns.get_indexer(test_df['movieId'].values)
        return self.pred_matrix[u_idx, m_idx]


def run_ablation_stages_2_to_4():
    print("Đang nạp dữ liệu chuẩn hóa từ Bước 1...")
    train_df, test_df, train_matrix, _, _, _, _ = load_and_preprocess_data()
    y_true = test_df['rating'].values

    print("\n" + "="*75)
    print("CHẠY THỰC NGHIỆM BƯỚC 3: KIỂM CHỨNG GIAI ĐOẠN 2 -> GIAI ĐOẠN 3 -> GIAI ĐOẠN 4")
    print("="*75)

    configs = [
        ("Giai đoạn 2: UBCF + Pearson + Z-Score", {"k_neighbors": 30, "use_sw": False, "use_iuf": False}),
        ("Giai đoạn 3: GĐ2 + Significance Weighting (Beta=30)", {"k_neighbors": 30, "use_sw": True, "beta": 30.0, "use_iuf": False}),
        ("Giai đoạn 4: GĐ3 + Inverse User Frequency (SW + IUF)", {"k_neighbors": 30, "use_sw": True, "beta": 30.0, "use_iuf": True}),
    ]

    for stage_name, params in configs:
        model = OptimizedUBCF(**params).fit(train_matrix)
        y_pred = model.predict_df(test_df)
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        extreme_pairs = int(np.sum(np.abs(model.sim_matrix) >= 0.999) // 2)

        print(f"\n[{stage_name}]")
        print(f"   - Sai số RMSE               : {rmse:.4f}")
        print(f"   - Sai số MAE                : {mae:.4f}")
        print(f"   - Thời gian chạy            : {model.train_time:.2f} giây")
        print(f"   - Số cặp hàng xóm ảo (|Sim|≈1): {extreme_pairs:,} cặp")


if __name__ == "__main__":
    run_ablation_stages_2_to_4()