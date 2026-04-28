from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def collect_image_paths(input_path: str) -> List[Path]:
    """
    입력 경로에서 이미지 파일 목록을 수집한다.

    input_path가 폴더이면 폴더 안의 이미지들을 파일명 기준으로 정렬한다.
    input_path가 파일 패턴이면 glob 패턴으로 이미지를 찾는다.

    예:
        data/
        data/*.jpg
    """
    path = Path(input_path)

    if path.is_dir():
        image_paths: List[Path] = []
        for extension in IMAGE_EXTENSIONS:
            image_paths.extend(path.glob(extension))
        return sorted(image_paths)

    return sorted(Path().glob(input_path))


def resize_keep_aspect(image: np.ndarray, max_width: int) -> np.ndarray:
    """
    이미지의 가로 길이가 max_width보다 크면 비율을 유지하며 축소한다.

    스마트폰 사진은 해상도가 커서 특징점 추출과 호모그래피 계산이 느릴 수 있다.
    그래서 기본적으로 가로 1200px 정도로 줄여서 처리한다.
    """
    if max_width <= 0:
        return image

    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_width = max_width
    new_height = int(height * scale)

    return cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)


def load_images(image_paths: List[Path], max_width: int) -> List[np.ndarray]:
    """
    이미지 파일들을 OpenCV BGR 이미지로 읽어온다.
    """
    images: List[np.ndarray] = []

    for image_path in image_paths:
        image = cv.imread(str(image_path))

        if image is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

        image = resize_keep_aspect(image, max_width)
        images.append(image)

    return images


def create_feature_detector(feature: str, nfeatures: int):
    """
    feature 이름에 맞는 특징점 검출기와 descriptor 방식을 생성한다.

    ORB:
        FAST 기반 특징점 검출 + BRIEF 기반 이진 descriptor.
        빠르고, Hamming distance로 매칭한다.

    SIFT:
        scale 변화와 rotation 변화에 강한 실수 descriptor.
        L2 distance로 매칭한다.
    """
    feature = feature.lower()

    if feature == "orb":
        detector = cv.ORB_create(nfeatures=nfeatures)
        norm_type = cv.NORM_HAMMING
        ratio = 0.75
        return detector, norm_type, ratio

    if feature == "sift":
        detector = cv.SIFT_create(nfeatures=nfeatures)
        norm_type = cv.NORM_L2
        ratio = 0.75
        return detector, norm_type, ratio

    raise ValueError(f"지원하지 않는 feature 방식입니다: {feature}")


def detect_and_compute(
    image: np.ndarray,
    feature: str,
    nfeatures: int,
):
    """
    한 이미지에서 keypoint와 descriptor를 추출한다.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    detector, _, _ = create_feature_detector(feature, nfeatures)

    keypoints, descriptors = detector.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) == 0:
        raise RuntimeError("특징점 또는 descriptor를 추출하지 못했습니다.")

    return keypoints, descriptors


def match_features(
    descriptors_src: np.ndarray,
    descriptors_dst: np.ndarray,
    feature: str,
    nfeatures: int,
) -> List[cv.DMatch]:
    """
    두 이미지의 descriptor를 매칭한다.

    src 이미지를 dst 이미지 좌표계로 보낼 것이므로,
    src descriptor와 dst descriptor를 비교한다.

    ORB는 binary descriptor이므로 Hamming distance를 사용한다.
    SIFT는 real-valued descriptor이므로 L2 distance를 사용한다.
    """
    _, norm_type, ratio = create_feature_detector(feature, nfeatures)

    matcher = cv.BFMatcher(norm_type)
    knn_matches = matcher.knnMatch(descriptors_src, descriptors_dst, k=2)

    good_matches: List[cv.DMatch] = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        best_match, second_best_match = pair

        if best_match.distance < ratio * second_best_match.distance:
            good_matches.append(best_match)

    return sorted(good_matches, key=lambda match: match.distance)


def save_match_visualization(
    src_image: np.ndarray,
    dst_image: np.ndarray,
    src_keypoints,
    dst_keypoints,
    matches: List[cv.DMatch],
    inlier_mask: np.ndarray | None,
    output_path: Path,
) -> None:
    """
    feature matching 결과 이미지를 저장한다.

    inlier_mask가 None이면 전체 매칭을 저장한다.
    inlier_mask가 있으면 RANSAC이 정상 매칭으로 판단한 inlier만 저장한다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if inlier_mask is None:
        visualization = cv.drawMatches(
            src_image,
            src_keypoints,
            dst_image,
            dst_keypoints,
            matches,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    else:
        visualization = cv.drawMatches(
            src_image,
            src_keypoints,
            dst_image,
            dst_keypoints,
            matches,
            None,
            matchesMask=inlier_mask.ravel().tolist(),
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    cv.imwrite(str(output_path), visualization)


def estimate_homography(
    src_image: np.ndarray,
    dst_image: np.ndarray,
    feature: str,
    nfeatures: int,
    ransac_threshold: float,
    debug_dir: Path,
    pair_name: str,
) -> Tuple[np.ndarray, dict]:
    """
    src_image를 dst_image 좌표계로 보내는 homography H를 계산한다.

    반환되는 H는 다음 의미를 가진다.

        dst_point ≈ H @ src_point

    즉, src 이미지의 점을 dst 이미지 좌표로 변환하는 행렬이다.
    """
    src_keypoints, src_descriptors = detect_and_compute(src_image, feature, nfeatures)
    dst_keypoints, dst_descriptors = detect_and_compute(dst_image, feature, nfeatures)

    matches = match_features(src_descriptors, dst_descriptors, feature, nfeatures)

    if len(matches) < 4:
        raise RuntimeError(
            f"{pair_name}: 매칭점이 4개 미만입니다. "
            f"현재 매칭점 수: {len(matches)}"
        )

    src_points = np.float32(
        [src_keypoints[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)

    dst_points = np.float32(
        [dst_keypoints[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)

    save_match_visualization(
        src_image,
        dst_image,
        src_keypoints,
        dst_keypoints,
        matches,
        None,
        debug_dir / f"{pair_name}_all_matches.jpg",
    )

    homography, inlier_mask = cv.findHomography(
        src_points,
        dst_points,
        cv.RANSAC,
        ransac_threshold,
    )

    if homography is None or inlier_mask is None:
        raise RuntimeError(f"{pair_name}: homography 계산에 실패했습니다.")

    inlier_count = int(inlier_mask.sum())

    if inlier_count < 4:
        raise RuntimeError(
            f"{pair_name}: RANSAC inlier가 4개 미만입니다. "
            f"현재 inlier 수: {inlier_count}"
        )

    save_match_visualization(
        src_image,
        dst_image,
        src_keypoints,
        dst_keypoints,
        matches,
        inlier_mask,
        debug_dir / f"{pair_name}_ransac_inliers.jpg",
    )

    stats = {
        "pair": pair_name,
        "src_keypoints": len(src_keypoints),
        "dst_keypoints": len(dst_keypoints),
        "matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": inlier_count / len(matches),
    }

    return homography, stats


def compute_pairwise_homographies(
    images: List[np.ndarray],
    feature: str,
    nfeatures: int,
    ransac_threshold: float,
    debug_dir: Path,
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    인접한 이미지 쌍 사이의 homography를 계산한다.

    pairwise_homographies[i]는 다음 의미를 가진다.

        image i+1 -> image i

    즉, i+1번째 이미지를 i번째 이미지 좌표계로 보내는 행렬이다.
    """
    pairwise_homographies: List[np.ndarray] = []
    all_stats: List[dict] = []

    for index in range(len(images) - 1):
        src_index = index + 1
        dst_index = index

        pair_name = f"image_{src_index + 1:02d}_to_image_{dst_index + 1:02d}"

        homography, stats = estimate_homography(
            src_image=images[src_index],
            dst_image=images[dst_index],
            feature=feature,
            nfeatures=nfeatures,
            ransac_threshold=ransac_threshold,
            debug_dir=debug_dir,
            pair_name=pair_name,
        )

        pairwise_homographies.append(homography)
        all_stats.append(stats)

    return pairwise_homographies, all_stats


def compute_global_transforms(
    pairwise_homographies: List[np.ndarray],
    image_count: int,
) -> List[np.ndarray]:
    """
    모든 이미지를 기준 이미지 좌표계로 보내는 global transform을 계산한다.

    기준 이미지는 가운데 이미지로 잡는다.
    이렇게 하면 왼쪽 끝 이미지를 기준으로 잡는 것보다 전체 왜곡이 줄어드는 편이다.
    """
    anchor_index = image_count // 2

    transforms: List[np.ndarray | None] = [None] * image_count
    transforms[anchor_index] = np.eye(3, dtype=np.float64)

    # 기준 이미지의 왼쪽 이미지들
    # pairwise_homographies[i]는 image i+1 -> image i 이므로,
    # image i -> image i+1 변환은 inverse(pairwise_homographies[i])이다.
    for index in range(anchor_index - 1, -1, -1):
        transforms[index] = transforms[index + 1] @ np.linalg.inv(
            pairwise_homographies[index]
        )

    # 기준 이미지의 오른쪽 이미지들
    for index in range(anchor_index + 1, image_count):
        transforms[index] = transforms[index - 1] @ pairwise_homographies[index - 1]

    final_transforms: List[np.ndarray] = []

    for transform in transforms:
        if transform is None:
            raise RuntimeError("global transform 계산 중 문제가 발생했습니다.")

        transform = transform / transform[2, 2]
        final_transforms.append(transform)

    return final_transforms


def compute_canvas(
    images: List[np.ndarray],
    transforms: List[np.ndarray],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    모든 이미지를 담을 수 있는 canvas 크기와 translation matrix를 계산한다.

    homography를 적용하면 이미지 좌표가 음수가 될 수 있다.
    따라서 전체 좌표를 양수 영역으로 옮기기 위해 translation matrix를 추가한다.
    """
    all_corners = []

    for image, transform in zip(images, transforms):
        height, width = image.shape[:2]

        corners = np.float32(
            [
                [0, 0],
                [width, 0],
                [width, height],
                [0, height],
            ]
        ).reshape(-1, 1, 2)

        warped_corners = cv.perspectiveTransform(corners, transform)
        all_corners.append(warped_corners)

    all_corners = np.concatenate(all_corners, axis=0)

    min_x, min_y = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    canvas_width = int(max_x - min_x)
    canvas_height = int(max_y - min_y)

    if canvas_width <= 0 or canvas_height <= 0:
        raise RuntimeError("canvas 크기가 올바르지 않습니다.")

    translation = np.array(
        [
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    return translation, (canvas_width, canvas_height)


def create_feather_weight(mask: np.ndarray, blend_width: int) -> np.ndarray:
    """
    feather blending에 사용할 weight map을 만든다.

    이미지 내부 중앙부는 weight가 크고,
    이미지 경계로 갈수록 weight가 작아지도록 만든다.
    이 방식을 사용하면 이미지가 겹치는 부분의 경계가 조금 더 자연스러워진다.
    """
    binary_mask = (mask > 0).astype(np.uint8)

    if blend_width <= 0:
        return binary_mask.astype(np.float32)

    distance = cv.distanceTransform(binary_mask, cv.DIST_L2, 3)
    weight = np.clip(distance / blend_width, 0.0, 1.0)

    return weight.astype(np.float32)


def blend_warped_images(
    images: List[np.ndarray],
    transforms: List[np.ndarray],
    translation: np.ndarray,
    canvas_size: Tuple[int, int],
    blend_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    모든 이미지를 canvas에 warp한 뒤 feather blending으로 합성한다.
    """
    canvas_width, canvas_height = canvas_size

    accumulator = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    for image, transform in zip(images, transforms):
        final_transform = translation @ transform

        warped_image = cv.warpPerspective(
            image,
            final_transform,
            (canvas_width, canvas_height),
        )

        source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)

        warped_mask = cv.warpPerspective(
            source_mask,
            final_transform,
            (canvas_width, canvas_height),
        )

        weight = create_feather_weight(warped_mask, blend_width)

        accumulator += warped_image.astype(np.float32) * weight[..., np.newaxis]
        weight_sum += weight

    safe_weight_sum = np.maximum(weight_sum, 1e-6)
    panorama = accumulator / safe_weight_sum[..., np.newaxis]
    panorama[weight_sum <= 1e-6] = 0

    return np.clip(panorama, 0, 255).astype(np.uint8), weight_sum


def crop_valid_region(image: np.ndarray, weight_sum: np.ndarray) -> np.ndarray:
    """
    실제 이미지가 존재하는 영역만 남기고 검은 테두리를 자른다.
    """
    valid_mask = (weight_sum > 1e-6).astype(np.uint8)

    x, y, width, height = cv.boundingRect(valid_mask)

    if width == 0 or height == 0:
        return image

    return image[y : y + height, x : x + width]


def stitch_images(
    images: List[np.ndarray],
    feature: str,
    nfeatures: int,
    ransac_threshold: float,
    blend_width: int,
    debug_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    전체 이미지 스티칭 파이프라인을 실행한다.
    """
    pairwise_homographies, stats = compute_pairwise_homographies(
        images=images,
        feature=feature,
        nfeatures=nfeatures,
        ransac_threshold=ransac_threshold,
        debug_dir=debug_dir,
    )

    transforms = compute_global_transforms(
        pairwise_homographies=pairwise_homographies,
        image_count=len(images),
    )

    translation, canvas_size = compute_canvas(images, transforms)

    panorama_raw, weight_sum = blend_warped_images(
        images=images,
        transforms=transforms,
        translation=translation,
        canvas_size=canvas_size,
        blend_width=blend_width,
    )

    panorama_cropped = crop_valid_region(panorama_raw, weight_sum)

    return panorama_raw, panorama_cropped, stats


def print_stats(stats: List[dict]) -> None:
    """
    매칭과 RANSAC 결과를 터미널에 출력한다.
    """
    print("\n[Matching Statistics]")

    for item in stats:
        print(f"- {item['pair']}")
        print(f"  src keypoints : {item['src_keypoints']}")
        print(f"  dst keypoints : {item['dst_keypoints']}")
        print(f"  matches       : {item['matches']}")
        print(f"  inliers       : {item['inliers']}")
        print(f"  inlier ratio  : {item['inlier_ratio']:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatic panorama image stitching using feature matching and homography estimation."
    )

    parser.add_argument(
        "--input",
        default="data",
        help="입력 이미지 폴더 또는 glob 패턴. 예: data 또는 data/*.jpg",
    )

    parser.add_argument(
        "--output",
        default="results/stitched_panorama.jpg",
        help="최종 스티칭 결과 이미지 저장 경로",
    )

    parser.add_argument(
        "--feature",
        default="orb",
        choices=["orb", "sift"],
        help="특징점 및 descriptor 방식",
    )

    parser.add_argument(
        "--nfeatures",
        type=int,
        default=5000,
        help="최대 특징점 개수",
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=1200,
        help="처리 전 이미지 최대 가로 크기. 0이면 resize하지 않음",
    )

    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=5.0,
        help="RANSAC reprojection threshold",
    )

    parser.add_argument(
        "--blend-width",
        type=int,
        default=40,
        help="feather blending 경계 부드러움 정도",
    )

    parser.add_argument(
        "--debug-dir",
        default="results/debug",
        help="매칭 시각화 결과 저장 폴더",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_paths = collect_image_paths(args.input)

    if len(image_paths) < 2:
        raise RuntimeError(
            "이미지가 2장 이상 필요합니다. "
            "과제 조건을 만족하려면 data 폴더에 3장 이상의 이미지를 넣는 것을 추천합니다."
        )

    if len(image_paths) < 3:
        print(
            "[Warning] 현재 이미지는 2장입니다. "
            "과제 요구사항은 3장 이상의 이미지 또는 비디오입니다."
        )

    print("[Input Images]")
    for image_path in image_paths:
        print(f"- {image_path}")

    images = load_images(image_paths, args.max_width)

    debug_dir = Path(args.debug_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panorama_raw, panorama_cropped, stats = stitch_images(
        images=images,
        feature=args.feature,
        nfeatures=args.nfeatures,
        ransac_threshold=args.ransac_threshold,
        blend_width=args.blend_width,
        debug_dir=debug_dir,
    )

    raw_output_path = output_path.with_name(output_path.stem + "_raw.jpg")

    cv.imwrite(str(raw_output_path), panorama_raw)
    cv.imwrite(str(output_path), panorama_cropped)

    print_stats(stats)

    print("\n[Output]")
    print(f"- raw panorama     : {raw_output_path}")
    print(f"- cropped panorama : {output_path}")
    print(f"- debug results    : {debug_dir}")


if __name__ == "__main__":
    main()