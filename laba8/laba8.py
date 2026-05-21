from pathlib import Path
from io import BytesIO
import csv
import requests
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# НАСТРОЙКИ
# -----------------------------
ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"

LIMIT = 5

OUTPUT_DIR = Path("lab8_output")
INPUT_DIR = OUTPUT_DIR / "input_png"
GRAY_BEFORE_DIR = OUTPUT_DIR / "grayscale_before"
GRAY_AFTER_DIR = OUTPUT_DIR / "grayscale_after"
COLOR_AFTER_DIR = OUTPUT_DIR / "contrasted_color"
HIST_DIR = OUTPUT_DIR / "histograms"
NGLDM_VIS_DIR = OUTPUT_DIR / "ngldm_visualization"
NGLDM_CSV_DIR = OUTPUT_DIR / "ngldm_csv"
DEMO_DIR = OUTPUT_DIR / "demos"

FEATURES_CSV_PATH = OUTPUT_DIR / "features.csv"

for folder in [
    OUTPUT_DIR,
    INPUT_DIR,
    GRAY_BEFORE_DIR,
    GRAY_AFTER_DIR,
    COLOR_AFTER_DIR,
    HIST_DIR,
    NGLDM_VIS_DIR,
    NGLDM_CSV_DIR,
    DEMO_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

NGLDM_D = 2

N_LEVELS = 16

ALPHA = 1

USE_LOG_FOR_VIS = True

GAMMA = 0.7


# -----------------------------
# ЗАГРУЗКА ИЗОБРАЖЕНИЙ ИЗ API
# -----------------------------
def fetch_sample_image_urls(origin: str, sample_id: str, limit: int | None = None) -> list[str]:
    response = requests.get(f"{origin}/api/samples/{sample_id}", timeout=30)
    response.raise_for_status()
    sample_data = response.json()

    image_urls = [f"{origin}/images/{page['filename']}" for page in sample_data["pages"]]

    if limit is not None:
        image_urls = image_urls[:limit]

    return image_urls


def download_images_as_png(image_urls: list[str], out_dir: Path) -> list[Path]:
    saved_paths = []

    for idx, url in enumerate(image_urls, start=1):
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
        out_path = out_dir / f"img_{idx:03d}.png"
        img.save(out_path, format="PNG")
        saved_paths.append(out_path)

        print(f"[OK] Скачано: {out_path}")

    return saved_paths


# -----------------------------
# RGB <-> HSL
# -----------------------------
def rgb_to_hsl_vectorized(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RGB uint8 -> H, S, L в диапазоне [0, 1]
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    l = (maxc + minc) / 2.0

    s = np.zeros_like(l, dtype=np.float32)
    nonzero = delta > 1e-8
    s[nonzero] = delta[nonzero] / (1.0 - np.abs(2.0 * l[nonzero] - 1.0) + 1e-12)

    h = np.zeros_like(l, dtype=np.float32)

    mask_r = nonzero & (maxc == r)
    mask_g = nonzero & (maxc == g)
    mask_b = nonzero & (maxc == b)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-12)) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-12)) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-12)) + 4.0

    h = h / 6.0
    h = np.mod(h, 1.0)

    return h, s, l


def hsl_to_rgb_vectorized(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    """
    H, S, L в диапазоне [0, 1] -> RGB uint8
    """
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hp = h * 6.0
    x = c * (1.0 - np.abs((hp % 2.0) - 1.0))

    r1 = np.zeros_like(h, dtype=np.float32)
    g1 = np.zeros_like(h, dtype=np.float32)
    b1 = np.zeros_like(h, dtype=np.float32)

    m0 = (0.0 <= hp) & (hp < 1.0)
    m1 = (1.0 <= hp) & (hp < 2.0)
    m2 = (2.0 <= hp) & (hp < 3.0)
    m3 = (3.0 <= hp) & (hp < 4.0)
    m4 = (4.0 <= hp) & (hp < 5.0)
    m5 = (5.0 <= hp) & (hp < 6.0)

    r1[m0], g1[m0], b1[m0] = c[m0], x[m0], 0.0
    r1[m1], g1[m1], b1[m1] = x[m1], c[m1], 0.0
    r1[m2], g1[m2], b1[m2] = 0.0, c[m2], x[m2]
    r1[m3], g1[m3], b1[m3] = 0.0, x[m3], c[m3]
    r1[m4], g1[m4], b1[m4] = x[m4], 0.0, c[m4]
    r1[m5], g1[m5], b1[m5] = c[m5], 0.0, x[m5]

    m = l - c / 2.0

    r = np.clip((r1 + m) * 255.0, 0, 255).astype(np.uint8)
    g = np.clip((g1 + m) * 255.0, 0, 255).astype(np.uint8)
    b = np.clip((b1 + m) * 255.0, 0, 255).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def lightness_to_gray_uint8(l: np.ndarray) -> np.ndarray:
    return np.clip(np.round(l * 255.0), 0, 255).astype(np.uint8)


# -----------------------------
# СТЕПЕННОЕ ПРЕОБРАЗОВАНИЕ ЯРКОСТИ
# -----------------------------
def power_contrast_channel(channel: np.ndarray, gamma: float = GAMMA) -> tuple[np.ndarray, float, float]:
    """
    Степенное преобразование яркости для канала L.

    Формула: L_out = L_in ** gamma, где L_in находится в диапазоне [0, 1].
    При gamma < 1 изображение становится светлее, при gamma > 1 — темнее.
    """
    fmin = float(channel.min())
    fmax = float(channel.max())

    if gamma <= 0:
        raise ValueError("Параметр gamma должен быть больше 0")

    out = np.power(np.clip(channel, 0.0, 1.0), gamma)
    out = np.clip(out, 0.0, 1.0)
    return out, fmin, fmax


# -----------------------------
# NGLDM
# -----------------------------
def quantize_gray(gray: np.ndarray, n_levels: int = 16) -> np.ndarray:
    """
    Квантование gray uint8 -> уровни 0..n_levels-1
    """
    q = (gray.astype(np.int32) * n_levels) // 256
    q = np.clip(q, 0, n_levels - 1)
    return q.astype(np.int32)


def compute_ngldm(gray: np.ndarray, n_levels: int = 16, d: int = 2, alpha: int = 1) -> np.ndarray:
    """
    Строит матрицу NGLDM.

    Для каждого пикселя:
    - берём его квантованный уровень i;
    - считаем количество соседей в окрестности радиуса d,
      у которых |neighbor - center| <= alpha;
    - dependence size j = 1 + число таких соседей;
    - увеличиваем элемент matrix[i, j-1].
    """
    q = quantize_gray(gray, n_levels=n_levels).astype(np.int16)

    h, w = q.shape
    max_dep = (2 * d + 1) ** 2

    dependence = np.ones((h, w), dtype=np.int32)

    sentinel = -10_000
    padded = np.pad(q, d, mode="constant", constant_values=sentinel)

    for dy in range(-d, d + 1):
        for dx in range(-d, d + 1):
            if dy == 0 and dx == 0:
                continue

            neigh = padded[d + dy:d + dy + h, d + dx:d + dx + w]
            valid = neigh != sentinel
            similar = valid & (np.abs(neigh - q) <= alpha)
            dependence += similar.astype(np.int32)

    matrix = np.zeros((n_levels, max_dep), dtype=np.int32)

    q_flat = q.ravel()
    dep_flat = dependence.ravel()

    np.add.at(matrix, (q_flat, dep_flat - 1), 1)

    return matrix


def compute_sne_lne(ngldm: np.ndarray) -> tuple[float, float]:
    """
    SNE = Small Number Emphasis
    LNE = Large Number Emphasis
    """
    total = float(ngldm.sum())
    if total == 0:
        return 0.0, 0.0

    j = np.arange(1, ngldm.shape[1] + 1, dtype=np.float64)

    sne = float(np.sum(ngldm / (j[None, :] ** 2)) / total)
    lne = float(np.sum(ngldm * (j[None, :] ** 2)) / total)

    return sne, lne


# -----------------------------
# СОХРАНЕНИЕ
# -----------------------------
def save_gray_image(img: np.ndarray, path: Path) -> None:
    Image.fromarray(img, mode="L").save(path)


def save_rgb_image(img: np.ndarray, path: Path) -> None:
    Image.fromarray(img, mode="RGB").save(path)


def save_matrix_csv(matrix: np.ndarray, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        for row in matrix:
            writer.writerow(row.tolist())


def save_ngldm_visualization(matrix: np.ndarray, out_path: Path, title: str, use_log: bool = True) -> None:
    vis = matrix.astype(np.float64)

    if use_log and np.max(vis) > 0:
        vis = np.log1p(vis)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(vis, cmap="gray", aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Размер зависимости j")
    ax.set_ylabel("Уровень серого i")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_histograms(gray_before: np.ndarray, gray_after: np.ndarray, out_path: Path, title: str) -> None:
    hist_before, _ = np.histogram(gray_before.ravel(), bins=256, range=(0, 256))
    hist_after, _ = np.histogram(gray_after.ravel(), bins=256, range=(0, 256))
    x = np.arange(256)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].bar(x, hist_before, width=1.0)
    axes[0].set_title(f"{title} — до контрастирования")
    axes[0].set_xlabel("Яркость")
    axes[0].set_ylabel("Количество пикселей")
    axes[0].set_xlim(0, 255)

    axes[1].bar(x, hist_after, width=1.0)
    axes[1].set_title(f"{title} — после контрастирования")
    axes[1].set_xlabel("Яркость")
    axes[1].set_ylabel("Количество пикселей")
    axes[1].set_xlim(0, 255)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_demo(
    rgb_before: np.ndarray,
    gray_before: np.ndarray,
    rgb_after: np.ndarray,
    gray_after: np.ndarray,
    out_path: Path,
    title: str
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(rgb_before)
    axes[0, 0].set_title("Исходное цветное")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray_before, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Исходное полутоновое")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(rgb_after)
    axes[1, 0].set_title("Контрастированное цветное")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gray_after, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Контрастированное полутоновое")
    axes[1, 1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# ОБРАБОТКА ОДНОГО ИЗОБРАЖЕНИЯ
# -----------------------------
def process_one_image(path: Path) -> dict:
    stem = path.stem

    rgb_before = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    # 1. RGB to HSL
    h, s, l_before = rgb_to_hsl_vectorized(rgb_before)
    gray_before = lightness_to_gray_uint8(l_before)

    # 2. NGLDM и признаки ДО
    ngldm_before = compute_ngldm(gray_before, n_levels=N_LEVELS, d=NGLDM_D, alpha=ALPHA)
    sne_before, lne_before = compute_sne_lne(ngldm_before)

    # 3. Степенное преобразование яркости канала L
    l_after, lmin_before, lmax_before = power_contrast_channel(l_before, gamma=GAMMA)
    gray_after = lightness_to_gray_uint8(l_after)

    # 4. Обратно в RGB
    rgb_after = hsl_to_rgb_vectorized(h, s, l_after)

    # 5. NGLDM и признаки ПОСЛЕ
    ngldm_after = compute_ngldm(gray_after, n_levels=N_LEVELS, d=NGLDM_D, alpha=ALPHA)
    sne_after, lne_after = compute_sne_lne(ngldm_after)

    # 6. Сохранение изображений
    gray_before_path = GRAY_BEFORE_DIR / f"{stem}_gray_before.png"
    gray_after_path = GRAY_AFTER_DIR / f"{stem}_gray_after.png"
    color_after_path = COLOR_AFTER_DIR / f"{stem}_contrasted_color.png"
    hist_path = HIST_DIR / f"{stem}_histograms.png"

    ngldm_before_csv_path = NGLDM_CSV_DIR / f"{stem}_ngldm_before.csv"
    ngldm_after_csv_path = NGLDM_CSV_DIR / f"{stem}_ngldm_after.csv"
    ngldm_before_vis_path = NGLDM_VIS_DIR / f"{stem}_ngldm_before.png"
    ngldm_after_vis_path = NGLDM_VIS_DIR / f"{stem}_ngldm_after.png"

    demo_path = DEMO_DIR / f"{stem}_demo.png"

    save_gray_image(gray_before, gray_before_path)
    save_gray_image(gray_after, gray_after_path)
    save_rgb_image(rgb_after, color_after_path)

    save_histograms(gray_before, gray_after, hist_path, title=stem)

    save_matrix_csv(ngldm_before, ngldm_before_csv_path)
    save_matrix_csv(ngldm_after, ngldm_after_csv_path)

    save_ngldm_visualization(
        ngldm_before,
        ngldm_before_vis_path,
        title=f"{stem} — NGLDM до контрастирования",
        use_log=USE_LOG_FOR_VIS
    )
    save_ngldm_visualization(
        ngldm_after,
        ngldm_after_vis_path,
        title=f"{stem} — NGLDM после контрастирования",
        use_log=USE_LOG_FOR_VIS
    )

    save_demo(rgb_before, gray_before, rgb_after, gray_after, demo_path, title=stem)

    print(f"[OK] Обработано: {path.name}")
    print(f"     gray before:   {gray_before_path}")
    print(f"     gray after:    {gray_after_path}")
    print(f"     color after:   {color_after_path}")
    print(f"     histograms:    {hist_path}")
    print(f"     ngldm before:  {ngldm_before_vis_path}")
    print(f"     ngldm after:   {ngldm_after_vis_path}")
    print(f"     demo:          {demo_path}")
    print(f"     SNE before:    {sne_before:.6f}")
    print(f"     LNE before:    {lne_before:.6f}")
    print(f"     SNE after:     {sne_after:.6f}")
    print(f"     LNE after:     {lne_after:.6f}")

    return {
        "image": path.name,
        "gray_before_path": str(gray_before_path),
        "gray_after_path": str(gray_after_path),
        "color_after_path": str(color_after_path),
        "hist_path": str(hist_path),
        "ngldm_before_csv": str(ngldm_before_csv_path),
        "ngldm_after_csv": str(ngldm_after_csv_path),
        "ngldm_before_vis": str(ngldm_before_vis_path),
        "ngldm_after_vis": str(ngldm_after_vis_path),
        "demo_path": str(demo_path),
        "lmin_before": round(lmin_before, 6),
        "lmax_before": round(lmax_before, 6),
        "sne_before": round(sne_before, 6),
        "lne_before": round(lne_before, 6),
        "sne_after": round(sne_after, 6),
        "lne_after": round(lne_after, 6),
    }


def save_features_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "image",
        "lmin_before",
        "lmax_before",
        "sne_before",
        "lne_before",
        "sne_after",
        "lne_after",
        "gray_before_path",
        "gray_after_path",
        "color_after_path",
        "hist_path",
        "ngldm_before_csv",
        "ngldm_after_csv",
        "ngldm_before_vis",
        "ngldm_after_vis",
        "demo_path",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=== Лабораторная работа №8 ===")
    print("Вариант 9: NGLDM, d=2, признаки SNE и LNE")
    print("Метод преобразования яркости: степенное преобразование")
    print(f"Параметр gamma: {GAMMA}")
    print()

    print("Загрузка изображений из API...")
    image_urls = fetch_sample_image_urls(ORIGIN, SAMPLE_ID, limit=LIMIT)
    print(f"Найдено изображений: {len(image_urls)}")

    input_paths = download_images_as_png(image_urls, INPUT_DIR)

    print("\nЗапуск обработки...")
    rows = []
    for path in input_paths:
        rows.append(process_one_image(path))

    save_features_csv(rows, FEATURES_CSV_PATH)

    print("\nГотово.")
    print(f"Результаты сохранены в: {OUTPUT_DIR.resolve()}")
    print(f"Таблица признаков:     {FEATURES_CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()