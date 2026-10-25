import time
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from step1_data_eda import load_and_preprocess_data


class ContentBasedBaseline:
    """
    GIAI ĐOẠN 1A (BASELINE 1): Content-Based Filtering (CBCF)
    """
    def __init__(self):
        self.pred_matrix = None
        self.user_index = None
        self.movie_columns = None
        self.train_time = 0.0

    def fit(self, train_matrix: pd.DataFrame, genre_matrix: pd.DataFrame):
        start_time = time.perf_counter()

        self.user_index = train_matrix.index
        self.movie_columns = train_matrix.columns

        R = train_matrix.fillna(0.0).values
        M = (~train_matrix.isna()).astype(np.float64).values

        user_counts = np.maximum(M.sum(axis=1, keepdims=True), 1.0)
        user_means = R.sum(axis=1, keepdims=True) / user_counts

        G = genre_matrix.values.astype(np.float64)
        genre_row_sums = G.sum(axis=1, keepdims=True)
        G_norm = np.divide(G, genre_row_sums, out=np.zeros_like(G), where=genre_row_sums > 0)

        user_genre_score_sum = R @ G_norm
        user_genre_weight_sum = M @ G_norm

        # Đã thêm out=np.zeros_like(...) để triệt tiêu hoàn toàn UserWarning trên Python 3.14
        genre_avg = np.divide(
            user_genre_score_sum,
            user_genre_weight_sum,
            out=np.zeros_like(user_genre_score_sum),
            where=user_genre_weight_sum > 0
        )
        P = np.where(user_genre_weight_sum > 0, genre_avg, user_means)

        pred = P @ G_norm.T
        no_genre_mask = (genre_row_sums.flatten() == 0)
        pred[:, no_genre_mask] = user_means

        self.pred_matrix = np.clip(pred, 0.5, 5.0)
        self.train_time = time.perf_counter() - start_time
        return self

    def predict_df(self, test_df: pd.DataFrame) -> np.ndarray:
        u_idx = self.user_index.get_indexer(test_df['userId'].values)
        m_idx = self.movie_columns.get_indexer(test_df['movieId'].values)
        return self.pred_matrix[u_idx, m_idx]


class RawCosineUBCF:
    """
    GIAI ĐOẠN 1B (BASELINE 2): User-Based Collaborative Filtering với Độ tương đồng Cosine thô
    (Đã tối ưu hóa Vectorization: chạy dưới 1.5 giây thay vì 83 giây)
    """
    def __init__(self, k_neighbors: int = 30):
        self.k = k_neighbors
        self.sim_matrix = None
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

        user_means = np.divide(R.sum(axis=1), np.maximum(M.sum(axis=1), 1.0))
        global_mean = R.sum() / M.sum()

        # 1. Tính Ma trận Tương đồng Cosine trên tập phim giao nhau
        numerator = R @ R.T
        R_sq = R ** 2
        sum_sq_u_on_common = R_sq @ M.T
        sum_sq_v_on_common = M @ R_sq.T
        denominator = np.sqrt(sum_sq_u_on_common * sum_sq_v_on_common)

        sim = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        np.fill_diagonal(sim, 0.0)
        self.sim_matrix = sim

        # 2. Dự đoán nhanh: Chỉ duyệt qua các cột phim có số người xem > K
        item_rater_counts = M.sum(axis=0).astype(int)
        pred = np.zeros((n_users, n_movies), dtype=np.float64)

        # Với các phim có <= K người xem: Toàn bộ những người đã xem đều nằm trong Top-K!
        # Do đó có thể nhân ma trận trực tiếp trong 0.01 giây:
        sparse_items = item_rater_counts <= self.k
        if np.any(sparse_items):
            R_sub = R[:, sparse_items]
            M_sub = M[:, sparse_items]
            numer_sub = sim @ R_sub
            denom_sub = np.abs(sim) @ M_sub
            pred_sub = np.divide(numer_sub, denom_sub, out=np.zeros_like(numer_sub), where=denom_sub > 0)
            pred[:, sparse_items] = np.where(denom_sub > 0, pred_sub, user_means[:, None])

        # Với các phim phổ biến (> K người xem, chỉ có khoảng ~1,100 phim): Lọc đúng Top-K người đã xem phim đó
        dense_item_indices = np.where(~sparse_items)[0]
        for j in dense_item_indices:
            raters = np.where(M[:, j] > 0)[0]
            sim_raters = sim[:, raters]        # (610 x n_raters)
            r_raters = R[raters, j]            # (n_raters,)

            top_k_local = np.argpartition(sim_raters, -self.k, axis=1)[:, -self.k:]
            w_top = np.take_along_axis(sim_raters, top_k_local, axis=1)
            r_top = r_raters[top_k_local]

            denom_j = np.abs(w_top).sum(axis=1)
            numer_j = (w_top * r_top).sum(axis=1)
            pred_j = np.divide(numer_j, denom_j, out=np.zeros_like(numer_j), where=denom_j > 0)
            pred[:, j] = np.where(denom_j > 0, pred_j, user_means)

        self.pred_matrix = np.clip(pred, 0.5, 5.0)
        self.train_time = time.perf_counter() - start_time
        return self

    def predict_df(self, test_df: pd.DataFrame) -> np.ndarray:
        u_idx = self.user_index.get_indexer(test_df['userId'].values)
        m_idx = self.movie_columns.get_indexer(test_df['movieId'].values)
        return self.pred_matrix[u_idx, m_idx]


if __name__ == "__main__":
    train_df, test_df, train_matrix, movies_indexed, genre_matrix, _, _ = load_and_preprocess_data()
    y_true = test_df['rating'].values

    cb = ContentBasedBaseline().fit(train_matrix, genre_matrix)
    print(f"[Baseline 1A - CBCF] RMSE: {root_mean_squared_error(y_true, cb.predict_df(test_df)):.4f} | Time: {cb.train_time:.2f}s")

    raw = RawCosineUBCF(k_neighbors=30).fit(train_matrix)
    print(f"[Baseline 1B - Raw Cosine] RMSE: {root_mean_squared_error(y_true, raw.predict_df(test_df)):.4f} | Time: {raw.train_time:.2f}s")