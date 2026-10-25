from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from step1_data_eda import load_and_preprocess_data
from step2_baselines import ContentBasedBaseline, RawCosineUBCF
from step3_ubcf_optimized import OptimizedUBCF

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def compute_beyond_accuracy_metrics(pred_matrix: np.ndarray, train_matrix: pd.DataFrame, top_n: int = 10, min_support: int = 3):
    """
    Tính toán Novelty@N (bits) và Catalog Coverage@N (%) trên các bộ phim ứng viên đủ độ tin cậy cộng đồng (>= min_support người xem)
    """
    M_mask = (~train_matrix.isna()).values
    n_users, n_movies = M_mask.shape

    item_pop = M_mask.sum(axis=0)
    item_self_info = np.log2((n_users + 1.0) / (item_pop + 1.0))

    # Loại bỏ phim user đã xem VÀ các phim quá rời rạc (< min_support người xem) khỏi danh sách xếp hạng Top-N
    invalid_candidates = M_mask | (item_pop[None, :] < min_support)
    scores_for_rec = np.where(invalid_candidates, -np.inf, pred_matrix)

    top_n_items_per_user = np.argsort(scores_for_rec, axis=1)[:, -top_n:][:, ::-1]

    rec_novelty_scores = item_self_info[top_n_items_per_user]
    mean_novelty = float(np.mean(rec_novelty_scores))

    eligible_movies_count = int(np.sum(item_pop >= min_support))
    unique_recommended_items = np.unique(top_n_items_per_user)
    coverage_pct = (len(unique_recommended_items) / max(eligible_movies_count, 1)) * 100.0

    return mean_novelty, coverage_pct, top_n_items_per_user


def get_top_recommendations_for_user(
    user_id: int,
    pred_matrix: np.ndarray,
    train_matrix: pd.DataFrame,
    movies_indexed: pd.DataFrame,
    top_n: int = 10,
    min_support: int = 3
) -> pd.DataFrame:
    """
    Trích xuất bảng danh sách Top-N bộ phim gợi ý chất lượng cao cho một người dùng cụ thể
    """
    u_idx = train_matrix.index.get_loc(user_id)
    M_mask = (~train_matrix.isna()).values
    item_pop = M_mask.sum(axis=0)

    user_scores = pred_matrix[u_idx, :].copy()
    user_scores[M_mask[u_idx, :] | (item_pop < min_support)] = -np.inf

    top_indices = np.argsort(user_scores)[::-1][:top_n]
    movie_ids = train_matrix.columns[top_indices]

    rec_df = pd.DataFrame({
        "Hạng": range(1, top_n + 1),
        "movieId": movie_ids,
        "Tên bộ phim (Title)": movies_indexed.loc[movie_ids, "title"].values,
        "Thể loại (Genres)": movies_indexed.loc[movie_ids, "genres"].values,
        "Điểm dự đoán": np.round(user_scores[top_indices], 3),
        "Số người đã xem (Pop)": item_pop[top_indices].astype(int)
    })
    return rec_df


def run_full_ablation_study():
    print("Đang nạp dữ liệu chuẩn hóa từ Bước 1...")
    train_df, test_df, train_matrix, movies_indexed, genre_matrix, _, _ = load_and_preprocess_data()
    y_true = test_df['rating'].values

    out_dir = Path("outputs")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("BƯỚC 4: CHẠY TỔNG HỢP NGHIÊN CỨU BÓC TÁCH (ABLATION STUDY - 5 MÔ HÌNH)")
    print("="*80)

    models_spec = [
        ("GĐ 1A: Content-Based (Genres)", ContentBasedBaseline(), "cbc"),
        ("GĐ 1B: UBCF Cosine Thô", RawCosineUBCF(k_neighbors=30), "ubcf"),
        ("GĐ 2: UBCF + Pearson Z-Score", OptimizedUBCF(k_neighbors=30, use_sw=False, use_iuf=False), "ubcf"),
        ("GĐ 3: GĐ2 + Sig. Weighting (β=30)", OptimizedUBCF(k_neighbors=30, use_sw=True, beta=30.0, use_iuf=False), "ubcf"),
        ("GĐ 4: GĐ3 + IUF (SW + IUF)", OptimizedUBCF(k_neighbors=30, use_sw=True, beta=30.0, use_iuf=True), "ubcf"),
    ]

    results = []
    trained_models = {}

    for name, model, m_type in models_spec:
        if m_type == "cbc":
            model.fit(train_matrix, genre_matrix)
            extreme_pairs = 0
        else:
            model.fit(train_matrix)
            extreme_pairs = int(np.sum(np.abs(model.sim_matrix) >= 0.999) // 2)

        y_pred = model.predict_df(test_df)
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        novelty, coverage, _ = compute_beyond_accuracy_metrics(model.pred_matrix, train_matrix, top_n=10)

        results.append({
            "Giai đoạn / Mô hình": name,
            "RMSE (↓)": round(rmse, 4),
            "MAE (↓)": round(mae, 4),
            "Novelty@10 (bits ↑)": round(novelty, 4),
            "Coverage@10 (% ↑)": round(coverage, 2),
            "Cặp hàng xóm ảo (|Sim|≈1)": extreme_pairs,
            "Thời gian (s)": round(model.train_time, 2)
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results)
    csv_path = out_dir / "ablation_results.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n=== BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM (ĐÃ LƯU RA outputs/ablation_results.csv) ===")
    print(results_df.to_string(index=False))

    sns.set_theme(style="whitegrid")
    short_labels = ["GĐ 1A\n(CBCF)", "GĐ 1B\n(Cosine Thô)", "GĐ 2\n(Pearson Z)", "GĐ 3\n(+SW β=30)", "GĐ 4\n(+SW +IUF)"]

    # Biểu đồ 1: RMSE & MAE
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(results_df))
    width = 0.36
    rects1 = ax1.bar(x - width/2, results_df["RMSE (↓)"], width, label="RMSE (Sai số toàn phương)", color="#1f77b4")
    rects2 = ax1.bar(x + width/2, results_df["MAE (↓)"], width, label="MAE (Sai số tuyệt đối)", color="#ff7f0e")
    ax1.set_title("Sự Cải Thiện Sai Số Dự Đoán (RMSE & MAE) Qua 4 Giai Đoạn Ablation Study", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=10)
    ax1.set_ylabel("Mức sai số (Sao - Càng thấp càng tốt)", fontsize=11)
    ax1.set_ylim(0.55, 1.05)
    ax1.legend(loc="upper right")
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.4f}", (rect.get_x() + rect.get_width()/2., h), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.4f}", (rect.get_x() + rect.get_width()/2., h), ha="center", va="bottom", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "eval_1_rmse_mae_progression.png", dpi=300)
    plt.close()

    # Biểu đồ 2: Novelty & Coverage
    ubcf_df = results_df.iloc[1:].copy()
    ubcf_labels = short_labels[1:]
    fig, ax_nov = plt.subplots(figsize=(8.5, 4.5))
    ax_cov = ax_nov.twinx()
    p1 = ax_nov.plot(ubcf_labels, ubcf_df["Novelty@10 (bits ↑)"], marker="o", linewidth=2.5, color="#2ca02c", label="Novelty@10 (Độ mới lạ - bits)")
    p2 = ax_cov.plot(ubcf_labels, ubcf_df["Coverage@10 (% ↑)"], marker="s", linewidth=2.5, linestyle="--", color="#d62728", label="Catalog Coverage@10 (%)")
    ax_nov.set_title("Đánh Giá Beyond-Accuracy: Tác Động Của SW Và IUF Lên Độ Mới Lạ & Độ Phủ", fontsize=12, fontweight="bold")
    ax_nov.set_ylabel("Novelty@10 (Bits thông tin)", color="#2ca02c", fontsize=10.5, fontweight="bold")
    ax_cov.set_ylabel("Catalog Coverage@10 (% Kho phim đủ chuẩn)", color="#d62728", fontsize=10.5, fontweight="bold")
    ax_cov.grid(False)
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax_nov.legend(lines, labels, loc="center right")
    plt.tight_layout()
    plt.savefig(fig_dir / "eval_2_beyond_accuracy_iuf.png", dpi=300)
    plt.close()

    # Biểu đồ 3: Độ nhạy Beta
    print("\nĐang quét tham số Beta (5 -> 100) để vẽ biểu đồ chứng minh lựa chọn Beta = 30...")
    betas = [5, 10, 20, 30, 40, 50, 75, 100]
    beta_rmses = []
    for b in betas:
        m_b = OptimizedUBCF(k_neighbors=30, use_sw=True, beta=b, use_iuf=True).fit(train_matrix)
        beta_rmses.append(root_mean_squared_error(y_true, m_b.predict_df(test_df)))

    plt.figure(figsize=(8, 4.2))
    plt.plot(betas, beta_rmses, marker="o", linewidth=2, color="#9467bd")
    best_idx = int(np.argmin(beta_rmses))
    plt.axvline(betas[best_idx], color="red", linestyle="--", label=f"Điểm tối ưu: Beta = {betas[best_idx]} (RMSE = {beta_rmses[best_idx]:.4f})")
    plt.title("Phân Tích Độ Nhạy Siêu Tham Số Ngưỡng Tin Cậy Beta (Significance Weighting)", fontsize=12, fontweight="bold")
    plt.xlabel("Ngưỡng số lượng phim xem chung tối thiểu (Beta)", fontsize=11)
    plt.ylabel("Sai số RMSE trên tập Test", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "eval_3_beta_sensitivity.png", dpi=300)
    plt.close()

    print(f"✅ Đã xuất 3 biểu đồ đánh giá thực nghiệm tại: {fig_dir.resolve()}")

    sample_user = 10
    print(f"\n=== CASE STUDY: ĐỐI CHIẾU TOP-5 GỢI Ý CHO USER #{sample_user} ===")
    print("\n1. Khi dùng [GĐ 1B: UBCF Cosine Thô]:")
    rec_raw = get_top_recommendations_for_user(sample_user, trained_models["GĐ 1B: UBCF Cosine Thô"].pred_matrix, train_matrix, movies_indexed, top_n=5)
    print(rec_raw.to_string(index=False))

    print("\n2. Khi dùng [GĐ 4: UBCF + Pearson Z-Score + SW + IUF] (Mô hình hoàn chỉnh):")
    rec_opt = get_top_recommendations_for_user(sample_user, trained_models["GĐ 4: GĐ3 + IUF (SW + IUF)"].pred_matrix, train_matrix, movies_indexed, top_n=5)
    print(rec_opt.to_string(index=False))


if __name__ == "__main__":
    run_full_ablation_study()