import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Cấu hình giao diện biểu đồ học thuật
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def load_and_preprocess_data(ratings_path="data/ratings.csv", movies_path="data/movies.csv", test_size=0.2, random_state=42):
    """
    1. Đọc dữ liệu gốc từ thư mục data/
    2. Làm sạch nhãn thể loại rác và đồng bộ hóa không gian người dùng - bộ phim
    3. Chia tập Train/Test (80/20) phân tầng (Stratified) theo từng người dùng
    """
    r_path, m_path = Path(ratings_path), Path(movies_path)
    if not r_path.exists() or not m_path.exists():
        raise FileNotFoundError("Không tìm thấy 'ratings.csv' hoặc 'movies.csv' trong thư mục 'data/'!")

    print("1. [DATA] Đang đọc và kiểm tra tính toàn vẹn của dữ liệu...")
    ratings = pd.read_csv(r_path)
    movies = pd.read_csv(m_path)

    # Làm sạch: Chỉ giữ lại 3 cột cốt lõi cho CF
    ratings_clean = ratings[['userId', 'movieId', 'rating']].copy()

    # Làm sạch cột genres cho Content-Based Baseline: Xóa nhãn '(no genres listed)'
    movies_clean = movies[['movieId', 'title', 'genres']].copy()
    movies_clean['genres_clean'] = movies_clean['genres'].replace('(no genres listed)', '')
    genre_matrix = movies_clean['genres_clean'].str.get_dummies(sep='|')
    
    # Nối thông tin tên phim và thể loại vào bảng đánh giá để phục vụ thống kê
    full_data = pd.merge(ratings_clean, movies_clean[['movieId', 'title', 'genres']], on='movieId', how='left')

    # Chia Train/Test (80/20) theo từng User để đảm bảo không user nào bị Cold-Start nhân tạo
    print("2. [SPLIT] Đang chia tập Train (80%) và Test (20%) phân tầng theo từng User...")
    train_df, test_df = train_test_split(
        ratings_clean,
        test_size=test_size,
        stratify=ratings_clean['userId'],
        random_state=random_state
    )

    # Tạo ma trận User-Item cố định kích thước (610 users x 9742 movies)
    all_users = np.sort(ratings_clean['userId'].unique())
    all_movies = np.sort(movies_clean['movieId'].unique())

    train_matrix = train_df.pivot(index='userId', columns='movieId', values='rating')
    train_matrix = train_matrix.reindex(index=all_users, columns=all_movies)

    # Đặt index cho genre_matrix theo đúng thứ tự all_movies
    movies_indexed = movies_clean.set_index('movieId').reindex(all_movies)
    genre_matrix.index = movies_clean['movieId']
    genre_matrix = genre_matrix.reindex(all_movies).fillna(0)

    # Thống kê tổng quan
    n_users = len(all_users)
    n_movies_total = len(all_movies)
    n_movies_rated = ratings_clean['movieId'].nunique()
    orphan_movies = n_movies_total - n_movies_rated
    sparsity = (1.0 - len(ratings_clean) / (n_users * n_movies_total)) * 100

    stats = {
        "n_users": n_users,
        "n_movies_total": n_movies_total,
        "n_movies_rated": n_movies_rated,
        "orphan_movies": orphan_movies,
        "n_ratings": len(ratings_clean),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "sparsity": sparsity,
        "n_genres": genre_matrix.shape[1]
    }

    print("\n=== BÁO CÁO LÀM SẠCH DỮ LIỆU (ML-LATEST-SMALL) ===")
    print(f"- Tổng số người dùng (Users)      : {n_users}")
    print(f"- Tổng số phim trong kho (Movies) : {n_movies_total} (Có điểm: {n_movies_rated} | Phim chưa ai xem: {orphan_movies})")
    print(f"- Tổng số lượt đánh giá (Ratings) : {stats['n_ratings']:,} (Train: {stats['n_train']:,} | Test: {stats['n_test']:,})")
    print(f"- Độ thưa ma trận (Sparsity)      : {sparsity:.2f}%")
    print(f"- Số thể loại phim độc lập (Clean): {stats['n_genres']} thể loại (Đã lọc bỏ '(no genres listed)')")

    return train_df, test_df, train_matrix, movies_indexed, genre_matrix, full_data, stats


def generate_eda_plots(ratings_df, output_dir="outputs/figures"):
    """
    Vẽ và xuất 3 biểu đồ EDA cốt lõi chứng minh cơ sở thực nghiệm cho 4 giai đoạn
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("\n3. [EDA] Đang xuất các biểu đồ phân tích dữ liệu sơ bộ...")

    # Biểu đồ 1: Phân phối điểm đánh giá (Cơ sở cho Z-Score Normalization)
    plt.figure(figsize=(8, 4.5))
    ax1 = sns.countplot(data=ratings_df, x='rating', hue='rating', palette='viridis', legend=False)
    plt.title("Phân phối Điểm đánh giá trên ml-latest-small (Cơ sở chuẩn hóa Z-Score)", fontsize=12, fontweight='bold')
    plt.xlabel("Mức điểm đánh giá (Sao)", fontsize=11)
    plt.ylabel("Số lượng lượt đánh giá", fontsize=11)
    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=8.5)
    plt.tight_layout()
    file1 = out_path / "eda_1_rating_distribution.png"
    plt.savefig(file1, dpi=300)
    plt.close()

    # Biểu đồ 2: Phân phối số lượng phim mỗi người dùng đánh giá (Cơ sở cho Significance Weighting)
    user_counts = ratings_df.groupby('userId')['movieId'].count()
    plt.figure(figsize=(8, 4.5))
    sns.histplot(user_counts, bins=40, kde=True, color='#2b5c8f')
    plt.axvline(user_counts.median(), color='red', linestyle='--', linewidth=1.5, label=f'Trung vị: {int(user_counts.median())} phim/user')
    plt.axvline(30, color='orange', linestyle='-', linewidth=1.5, label='Ngưỡng SW (Beta = 30)')
    plt.title("Phân phối Số lượng phim đánh giá của mỗi Người dùng", fontsize=12, fontweight='bold')
    plt.xlabel("Số lượng phim đã đánh giá", fontsize=11)
    plt.ylabel("Số lượng người dùng", fontsize=11)
    plt.legend()
    plt.tight_layout()
    file2 = out_path / "eda_2_user_activity.png"
    plt.savefig(file2, dpi=300)
    plt.close()

    # Biểu đồ 3: Đường cong Đuôi dài - Long Tail (Cơ sở cho trọng số IUF)
    item_counts = ratings_df.groupby('movieId')['userId'].count().sort_values(ascending=False).values
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(item_counts) + 1), item_counts, color='#d95f02', linewidth=2)
    plt.fill_between(range(1, len(item_counts) + 1), item_counts, color='#d95f02', alpha=0.2)
    plt.axhline(50, color='navy', linestyle='--', linewidth=1.2, label='Nhóm phim phổ biến (>50 người xem)')
    plt.title("Hiện tượng Đuôi dài (Long-Tail) của Kho phim (Cơ sở áp dụng IUF)", fontsize=12, fontweight='bold')
    plt.xlabel("Thứ hạng độ phổ biến của Bộ phim (Từ 1 đến 9,724)", fontsize=11)
    plt.ylabel("Số lượng người dùng đã xem", fontsize=11)
    plt.legend()
    plt.tight_layout()
    file3 = out_path / "eda_3_long_tail_items.png"
    plt.savefig(file3, dpi=300)
    plt.close()

    print(f"✅ Đã lưu 3 biểu đồ phân tích chất lượng cao (300 DPI) tại: {out_path.resolve()}")


if __name__ == "__main__":
    train_df, test_df, train_matrix, movies_indexed, genre_matrix, full_data, stats = load_and_preprocess_data()
    generate_eda_plots(full_data)