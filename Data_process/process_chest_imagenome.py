#!/usr/bin/env python3
"""
Batch process scene_graph JSON files with matching mimic images.
支持多进程并行处理以加速万级数据处理。
"""
import os
import sys
import os
import json
from PIL import Image, ImageDraw, ImageColor
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import re
from string import punctuation
import csv
from transformers.image_transforms import resize as hf_resize
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import time


# Paths
# SCENE_GRAPH_DIR = "/Users/tsn/Documents/Files/MyProgram/chest-imagenome-dataset-1.0.0/silver_dataset/scene_graph" #文件夹
#我需要将他修改为一个txt文件，里面存了很多json文件的路径
# SCENE_GRAPH_DIR = "/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/A-Chest-MINT-CoT/V-new/scene_graph_part_0.txt" #文件夹
SCENE_GRAPH_DIR = "/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/chest-imagenome-dataset-1.0.0/silver_dataset/scene_graph" #文件夹
MIMIC_IMG_DIR = "/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/mimic_cxr/files/"
OUTPUT_JSON_PATH = "/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/Data_process/dataset/chest_dataset_54880_full.json"
ALL_BBOX_REGIONS_PATH = "/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/Data_process/meta_data/all_bbox_names_and_regions_new.json"
MIMIC_JSON_PATH = "/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/Data_process/meta_data/annotation.json"
TAG_PATH = "/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/Data_process/meta_data/mimic-cxr-2.0.0-negbio.csv"

PATCH_SIZE = 14
MERGE_SIZE = 2 
FACTOR = PATCH_SIZE * MERGE_SIZE  # 28 (与 Qwen2.5-VL 图像处理器一致)
# Qwen2.5-VL 图像处理器的默认参数（与 transformers 中 Qwen2VLImageProcessor 保持一致）
# MIN_PIXELS = 56 * 56  # 3136 =32*32 和yaml的pixel保持一致
# MAX_PIXELS = 14 * 14 * 4 * 1280   #50176=224*224 M4内存受限的时候使用，和yaml中max_pixels保持一致
MIN_PIXELS = 32 * 32  # 3136 =32*32 和yaml的pixel保持一致
MAX_PIXELS = 54880 #589824 #768*768 #50176
THRESHOLD_RATIO = 0.5

def build_image_path_map(mi_josn_path: str, mimic_img_dir: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    id_to_full_path = {}  #样本id -> 图片绝对路径  
    id_to_image_path = {} #样本id —> 图片的不完整路径


    try:
        with open(mi_josn_path, "r", encoding="utf-8") as f:
            mi_josn_data = json.load(f)
        
        count = 0
        # Process train, val, and test splits
        #train:270790;  'val': 2130;  'test': 3858
        for split_key in ["train"]:
            if split_key not in mi_josn_data:
                continue

            records = mi_josn_data[split_key]
            print(f"Total records in split '{split_key}': {len(records)}")

            for record in tqdm(records, desc=f"Building image paths ({split_key})"):
                image_id = record.get("id", "")
                image_paths = record.get("image_path", [])

                if count == 0:
                    count += 1
                if image_id and image_paths:
                    if isinstance(image_paths, list):
                        id_to_image_path[image_id] = image_paths
                    else:
                        id_to_image_path[image_id] = [image_paths]
                    
                    # 图片路径加前缀，构建完整绝对路径
                    rel_path = image_paths[0] if isinstance(image_paths, list) else image_paths
                    full_path = os.path.join(mimic_img_dir, rel_path)
                    if count == 1:
                        count += 1
                    # Only add if file actually exists
                    if os.path.exists(full_path):
                        id_to_full_path[image_id] = full_path
        
        print(f"Built image path map: {len(id_to_full_path)} images found in annotation.json")
        return id_to_full_path, id_to_image_path
        
    except Exception as e:
        print(f"Error building image path map: {e}")
        return id_to_full_path, id_to_image_path



def load_all_bbox_and_regions(path: str) -> Tuple[set, Dict[str, List[str]]]:
    """Load bbox names and region mappings."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_bbox_names = set(data.get("all_bbox_names", []))
    regions = data.get("regions", {})
    return all_bbox_names, regions

def build_maps_from_sample(sample_data: dict) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Return (objects_map, attributes_map) keyed by bbox_name."""
    objects_map = {}
    for obj in sample_data.get("objects", []):
        name = obj.get("bbox_name")
        if not name:
            continue
        objects_map[name] = obj

    attributes_map = {}
    for attr in sample_data.get("attributes", []):
        name = attr.get("bbox_name")
        if not name:
            continue
        attributes_map[name] = attr
    return objects_map, attributes_map

def invert_regions(regions: Dict[str, List[str]]) -> Dict[str, str]:
    """Map bbox_name -> region_key"""
    out = {}
    for region_key, name_list in regions.items():
        for n in name_list:
            out[n] = region_key
    return out

def group_structures(sample_data: dict, regions: Dict[str, List[str]]) -> Dict[str, dict]:
    """Group structures by anatomical regions."""
    #将样本数据中的解剖结构按 7 个解剖区域分组
    objects_map, attributes_map = build_maps_from_sample(sample_data)
    b2r = invert_regions(regions)

    #将 objects_map 和 attributes_map 两个字典中所有的 bbox_name 键合并到一个集合 used_names 里面，去重后用于后续处理所有包含的解剖结构名字。
    used_names = set(objects_map.keys()) | set(attributes_map.keys())

    grouped: Dict[str, dict] = {}
    for name in sorted(used_names):
        region_key = b2r.get(name)
        if not region_key:
            continue
        grouped.setdefault(region_key, {"structures": []})

        struct_entry = {
            "name": name,
            "bbox": None,
            "descriptions": []
        }
        if name in objects_map:
            obj = objects_map[name]
            bbox = {}
            for k in [
                "original_x1", "original_y1", "original_x2", "original_y2",
                "original_width", "original_height"
            ]:
                if k in obj:
                    bbox[k] = obj[k]
            struct_entry["bbox"] = bbox if bbox else None
        if name in attributes_map:
            phrases = attributes_map[name].get("phrases", [])
            struct_entry["descriptions"] = phrases

            #######################

        grouped[region_key]["structures"].append(struct_entry)
    return grouped

def smart_resize(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
):
    """
    Qwen2-VL 官方 smart_resize 实现的对齐版本。
    保证：
      1. 高宽都是 factor 的倍数
      2. 总像素在 [min_pixels, max_pixels] 范围内
      3. 尽可能保持原始宽高比
    """
    # 与官方一致：极端长宽比直接报错
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    # 初始按 factor 对齐
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    # 如果总像素数超过 max_pixels，等比例缩小
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    # 如果总像素数小于 min_pixels，等比例放大
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar

def scale_bboxes_for_resized(structs_by_region: Dict[str, dict],
                             orig_w: int, orig_h: int,
                             new_w: int, new_h: int) -> None:
    """Scale bounding boxes from original to resized image dimensions.
    """
    sx = new_w / float(orig_w)
    sy = new_h / float(orig_h)
    for reg in structs_by_region.values():
        for s in reg.get("structures", []):
            bbox = s.get("bbox")
            if not bbox:
                continue
            if all(k in bbox for k in ("original_x1", "original_y1", "original_x2", "original_y2")):
                x1, y1, x2, y2 = bbox["original_x1"], bbox["original_y1"], bbox["original_x2"], bbox["original_y2"]
            else:
                continue

            # Direct stretch: different scale factors for x and y
            rx1 = int(round(x1 * sx))
            ry1 = int(round(y1 * sy))
            rx2 = int(round(x2 * sx))
            ry2 = int(round(y2 * sy))
            
            # Clamp to image bounds
            rx1 = max(0, min(new_w - 1, rx1))
            ry1 = max(0, min(new_h - 1, ry1))
            rx2 = max(0, min(new_w - 1, rx2))
            ry2 = max(0, min(new_h - 1, ry2))

            bbox["resized_x1"] = rx1
            bbox["resized_y1"] = ry1
            bbox["resized_x2"] = rx2
            bbox["resized_y2"] = ry2

def compute_region_patch_occupancy(structs_by_region: Dict[str, dict],
                                   img_w: int, img_h: int,
                                   patch_side: int = PATCH_SIZE * MERGE_SIZE,
                                   threshold_ratio: float = 0.5) -> Dict[str, List[int]]:
    """
    计算每个解剖区域占用了哪些 image patches，返回 patch ID 列表。
    
    使用与 Qwen2.5-VL 图像处理器相同的 patch_side (28 = patch_size * merge_size)。
    patch 编号从 0 开始，按行优先顺序 (row-major order) 计算。
    """
    occupied: Dict[str, List[int]] = {}
    patch_area = patch_side * patch_side
    grid_h = math.ceil(img_h / patch_side)
    grid_w = math.ceil(img_w / patch_side)

    for region_key, reg in structs_by_region.items():
        # union mask
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for s in reg.get("structures", []):
            bbox = s.get("bbox")
            if not bbox:
                continue
            if all(k in bbox for k in ("resized_x1", "resized_y1", "resized_x2", "resized_y2")):
                x1, y1, x2, y2 = bbox["resized_x1"], bbox["resized_y1"], bbox["resized_x2"], bbox["resized_y2"]
                #边界检查
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2 + 1, x1:x2 + 1] = 1
        region_patches: List[int] = []
        # 遍历所有patches（
        for r in range(grid_h):
            for c in range(grid_w):
                # 计算当前patch的像素范围
                px1, py1 = c * patch_side, r * patch_side
                px2, py2 = min(img_w, (c + 1) * patch_side), min(img_h, (r + 1) * patch_side)
                if px2 <= px1 or py2 <= py1:
                    continue
                # 提取这个patch区域的掩码
                sub = mask[py1:py2, px1:px2]
                if sub.size == 0:
                    continue
                # 如果重叠度 ≥ 50%，认为这个patch属于该区域
                if sub.sum() >= threshold_ratio * ((py2 - py1) * (px2 - px1)):
                    # Convert [r, c] to linear index: index = r * grid_w + c (0-based)
                    patch_index = r * grid_w + c # 转换为线性索引
                    region_patches.append(patch_index)
        # Deduplicate and sort
        occupied[region_key] = sorted(list(set(region_patches)))

    return occupied


def get_region_name_en(region_key: str) -> str:
    """Get English name for region."""
    region_names = {
        "Lung and Pleura": "Lung and Pleura",
        "Mediastinum and Airway": "Mediastinum and Airway",
        "Cardiac": "Cardiac",
        "Bones and Soft Tissue": "Bones and Soft Tissue",
        "Diaphragm and Abdomen": "Diaphragm and Abdomen"
    }
    return region_names.get(region_key, region_key.replace("_", " ").title())


def clean_report(report: Any) -> str:
    """
    Clean report text by removing redundant spaces and newline characters.

    - Normalize line breaks (CRLF/CR -> LF)
    - Strip each line and drop empty lines
    - Collapse whitespace to single spaces
    - Return a single-line string (safer for prompt formatting)
    """
    if report is None:
        return ""

    text = str(report)

    # Normalize common special whitespaces
    text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")  # NBSP
        .replace("\u200b", "")   # zero-width space
        .replace("\ufeff", "")   # BOM
    )

    # Clean per-line then join into one line
    lines: List[str] = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line.strip())
        if line:
            lines.append(line)

    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove spaces before common punctuation (optional but improves readability)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Clean up various formatting issues
    text = text.replace('. .', '.').replace('..', '.').replace('1. ', '') \
        .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
        .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ')
    # Remove multiple spaces (apply multiple times to handle cases with more than 2 spaces)
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text


def load_annotation_data(mi_json_path: str, image_id: str, 
                         tag_dict: Optional[Dict[str, int]] = None,
                         mi_cdrc_data: Optional[Dict] = None) -> Tuple[str, str]:
    """Load tag and report from MI_CDRC.json for given image_id.
    Returns: (impression_status, report_text)
    - impression_status: "normal" if tag==0, "abnormal" otherwise
    - report_text: the report field from MI_CDRC.json (restored to proper format)
    
    Args:
        tag_dict: Pre-loaded tag dictionary (optional, for performance)
        mi_cdrc_data: Pre-loaded MIMIC JSON data (optional, for performance)
    """
    try:
        # Load tag_dict if not provided
        if tag_dict is None:
            tag_dict = {}
            with open(TAG_PATH, mode='r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    study_id = row['study_id'].strip() if row.get('study_id') else None
                    if study_id:
                        if row.get('No Finding') == '1.0':
                            tag_dict[study_id] = 0
                        else:
                            tag_dict[study_id] = 1
        
        # Load mi_cdrc_data if not provided
        if mi_cdrc_data is None:
            with open(mi_json_path, "r", encoding="utf-8") as f:
                mi_cdrc_data = json.load(f)
        
        # Search in train, va, and test splits
        for split_key in ["train"]:
            if split_key not in mi_cdrc_data:
                continue
            for record in mi_cdrc_data[split_key]:
                if record.get("id") == image_id:
                    study_id = str(record.get("study_id", "")).strip()
                    if study_id not in tag_dict:
                        # Try to find similar keys
                        similar_keys = [k for k in tag_dict.keys() if str(k).strip() == study_id]
                        if similar_keys:
                            study_id = similar_keys[0]
                    
                    tag = tag_dict.get(study_id)
                    report = record.get("report", "")
                    # Restore report format before returning
                    report = clean_report(report)
                    # Default to Abnormal if tag is None
                    impression = "Normal" if tag == 0 else "Abnormal"
                    return impression, report
        # If not found, return default values
        return "normal", ""
    except Exception as e:
        return "normal", ""


def generate_messages_interleave_ra(grouped: Dict[str, dict],
                                    image_id: str,
                                    mi_json_path: Optional[str] = None,
                                    tag_dict: Optional[Dict[str, int]] = None,
                                    mi_cdrc_data: Optional[Dict] = None) -> List[Dict[str, str]]:
    """Generate messages_interleave_ra from grouped structures."""
    impression_status = "normal"
    report_text = ""
    if mi_json_path and os.path.exists(mi_json_path):
        impression_status, report_text = load_annotation_data(
            mi_json_path, image_id, tag_dict=tag_dict, mi_cdrc_data=mi_cdrc_data
        )
    
    # Collect regions with non-empty descriptions
    regions_with_desc = []
    for region_key, reg in grouped.items():
        all_descriptions = []
        for struct in reg.get("structures", []):
            descs = struct.get("descriptions", [])
            if descs:
                # Clean each phrase before adding
                # cleaned_descs = [clean_phrase(desc) for desc in descs if clean_phrase(desc)]
                # all_descriptions.extend(cleaned_descs)
                all_descriptions.extend(descs)

        # 对于一个分区的所有描述去重和排序
        unique_descriptions = sorted(list(set(all_descriptions)))
        
        if unique_descriptions:  # Only include regions with descriptions
            patches = reg.get("patches", [])
            regions_with_desc.append({
                "region_key": region_key,
                "region_name_en": get_region_name_en(region_key),
                "descriptions": unique_descriptions,
                "patches": patches,
            })
    
    # 开始结构化构建数据
    concern_areas = [r["region_name_en"] for r in regions_with_desc]
    if concern_areas:
        global_impression = f"[Status: {impression_status}] Key regions: {', '.join(concern_areas)}."    
    else:
        global_impression = f"[Status: {impression_status}]."
    
    # Build assistant content with steps
    assistant_parts = [
        f"### Global Impression: {global_impression}\n\n",
        "### Let's observe step by step.\n\n"
    ]
    
    for step_idx, region_data in enumerate(regions_with_desc, start=1):
        region_name = region_data["region_name_en"]
        patches = region_data["patches"]
        descriptions = region_data["descriptions"]
        
        # Format patch IDs as comma-separated string
        patch_str = ",".join(map(str, patches)) if patches else ""
        
        # Combine descriptions into text (descriptions are already cleaned)
        # Remove trailing periods from each description before joining to avoid double periods
        # Replace newlines with spaces to avoid unwanted line breaks in the output
        cleaned_descriptions = [
            desc.replace('\n', ' ').replace('\r', ' ')
                .replace('1.', ' ').replace('2.', ' ').replace('3.', ' ').replace('4.', ' ').replace('5.', ' ')
                .replace('. .', '.').replace('..', '.').replace('1. ', '')
                .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ')
                .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ')
                .replace('  ', ' ').replace('   ', ' ')
                .rstrip('.').strip()
            for desc in descriptions if desc.strip()
        ]
        # Use ". " to join phrases (normal format: period follows word)
        desc_text = ". ".join(cleaned_descriptions)
        # Clean up multiple spaces that may result from replacing newlines
        desc_text = re.sub(r'\s+', ' ', desc_text).strip()
        if desc_text and not desc_text.endswith("."):
            desc_text += "."
        
        # Build step content: Step k -> <interleave> -> region name -> descriptions
        step_content = f"### Step {step_idx}: [Observation Target: {region_name}]\n<interleave>{patch_str}<interleave>\n{desc_text}\n\n"
        assistant_parts.append(step_content)
    
    # Add final conclusion with report
    final_conclusion = report_text
    assistant_parts.append(f"### The final conclusion is: {final_conclusion}")
    assistant_content = "".join(assistant_parts)
    
    # Build user content
    user_content = """<image>You are an expert Radiologist. Your task is to interpret the provided chest X-ray image following a standard clinical workflow: first give a global impression (Normal or Abnormal) and list key regions of concern, then observe specific anatomical regions in detail. For each step, explicitly state the [Observation Target], focus on the relevant image regions, and provide a clear clinical description. Finally, summarize all findings into a comprehensive diagnostic report.
Format your response exactly as follows, separated by ###:
### Global Impression: [Status: {Normal/Abnormal}], Key regions: {List of Regions}.
### Let's observe step by step.
### Step 1: [Observation Target: Region Name]
{Findings}
### Step 2: 
...
### The final conclusion is: {Diagnostic Report}"""

    
    # Build messages_interleave_ra
    messages_interleave_ra = [
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": assistant_content
        }
    ]
    
    return messages_interleave_ra

def draw_boxes(img: Image.Image, structs_by_region: Dict[str, dict], use_original: bool, color_map: Optional[Dict[str, str]] = None) -> Image.Image:
    """Draw bounding boxes on image with different colors for each region.
    The legend is drawn at the bottom with enlarged color boxes and text.
    """
    if color_map is None:
        color_map = {
            "Lung and Pleura": "red",
            "Mediastinum and Airway": "orange",
            "Cardiac": "magenta",
            "Bones and Soft Tissue": "cyan",
            "Diaphragm and Abdomen": "yellow",
        }
    img_w, img_h = img.size
    
    # Create output image with extra space at bottom for legend
    # Increased height for larger legend (200 pixels instead of 150)
    legend_height = 200
    out = Image.new("RGB", (img_w, img_h + legend_height), (255, 255, 255))
    out.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(out)
    
    # Draw bounding boxes on the original image part
    for region_key, reg in structs_by_region.items():
        color = color_map.get(region_key, "lime")
        for s in reg.get("structures", []):
            bbox = s.get("bbox")
            if not bbox:
                continue
            if use_original:
                keys = ("original_x1", "original_y1", "original_x2", "original_y2")
            else:
                keys = ("resized_x1", "resized_y1", "resized_x2", "resized_y2")
            if all(k in bbox for k in keys):
                x1, y1, x2, y2 = bbox[keys[0]], bbox[keys[1]], bbox[keys[2]], bbox[keys[3]]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # Draw legend at bottom with enlarged boxes and text
    # Get English names for regions
    region_names_en = {
        "Lung and Pleura": "Lung and Pleura",
        "Mediastinum and Airway": "Mediastinum and Airway",
        "Cardiac": "Cardiac",
        "Bones and Soft Tissue": "Bones and Soft Tissue",
        "Diaphragm and Abdomen": "Diaphragm and Abdomen"
    }
    
    y_start = img_h + 20
    x_start = 30
    # Enlarged color box size: 35x25 (was 20x15)
    box_w, box_h = 35, 25
    x = x_start
    y = y_start
    
    # Try to load a larger font if available
    try:
        from PIL import ImageFont
        # Try to use a larger default font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                font = ImageFont.load_default()
    except:
        font = None
    
    # Draw legend items
    for region_key in sorted(color_map.keys()):
        # Check if this region has any structures
        if region_key in structs_by_region and structs_by_region[region_key].get("structures"):
            # Wrap to next line if needed
            region_name = region_names_en.get(region_key, region_key.replace("_", " ").title())
            text_width = len(region_name) * 10  # Approximate width
            if x + box_w + text_width + 50 > img_w:
                x = x_start
                y += 40  # Larger spacing for enlarged boxes
            
            color = color_map.get(region_key, "lime")
            # Draw enlarged color box
            draw.rectangle([x, y, x + box_w, y + box_h], fill=color, outline=(0, 0, 0, 255), width=2)
            # Draw enlarged text
            if font:
                draw.text((x + box_w + 8, y + 2), region_name, fill=(0, 0, 0, 255), font=font)
            else:
                draw.text((x + box_w + 8, y + 2), region_name, fill=(0, 0, 0, 255))
            x += box_w + text_width + 60  # Larger spacing between items
    
    return out


def draw_region_patches_vis(resized_img: Image.Image, patch_side: int, grid_w: int,
                            region_patch_map: Dict[str, List[int]],
                            color_map: Dict[str, str], out_path: str) -> None:
    """Draw region patches on resized image, with special handling for overlapping patches."""
    img_w, img_h = resized_img.size
    # 计算纵向有多少个 patch 行（保证整张图都被网格覆盖）
    grid_h = math.ceil(img_h / patch_side)
    # Create overlay with transparency
    overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    
    # Build patch occupancy map: (r, c) -> list of region_keys
    # Convert linear index back to (r, c) for visualization
    patch_to_regions: Dict[Tuple[int, int], List[str]] = {}
    for region_key, patch_indices in region_patch_map.items():
        for patch_idx in patch_indices:
            # Convert linear index to (row, col): r = index // grid_w, c = index % grid_w
            r = patch_idx // grid_w
            c = patch_idx % grid_w
            key = (r, c)
            if key not in patch_to_regions:
                patch_to_regions[key] = []
            patch_to_regions[key].append(region_key)
    
    # Draw patches：遍历整张图的所有 patch 网格
    for r in range(grid_h):
        for c in range(grid_w):
            x1, y1 = c * patch_side, r * patch_side
            x2, y2 = min(img_w, (c + 1) * patch_side), min(img_h, (r + 1) * patch_side)
            if x2 <= x1 or y2 <= y1:
                continue

            key = (r, c)
            region_keys = patch_to_regions.get(key, [])

            if not region_keys:
                # 不属于任何解剖分区的 patch：只画一个浅色网格框
                draw.rectangle(
                    [x1, y1, x2 - 1, y2 - 1],
                    outline=(200, 200, 200, 255),
                    width=1,
                )
            elif len(region_keys) == 1:
                # Single region: use its color with transparency
                region_key = region_keys[0]
                color = color_map.get(region_key, "lime")
                rgb = ImageColor.getrgb(color)
                fill = (rgb[0], rgb[1], rgb[2], 100)
                outline = (rgb[0], rgb[1], rgb[2], 255)
                draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=fill, outline=outline, width=3)
            else:
                # Multiple regions: use diagonal stripes or checkerboard pattern
                # Draw each region's color in different quadrants/corners
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                for i, region_key in enumerate(region_keys):
                    color = color_map.get(region_key, "lime")
                    rgb = ImageColor.getrgb(color)
                    # Divide patch into diagonal halves for each region
                    if i == 0:
                        # Top-left triangle
                        draw.polygon(
                            [(x1, y1), (x2, y1), (x1, y2)],
                            fill=(rgb[0], rgb[1], rgb[2], 120),
                            outline=(rgb[0], rgb[1], rgb[2], 255),
                        )
                    elif i == 1:
                        # Bottom-right triangle
                        draw.polygon(
                            [(x2, y1), (x2, y2), (x1, y2)],
                            fill=(rgb[0], rgb[1], rgb[2], 120),
                            outline=(rgb[0], rgb[1], rgb[2], 255),
                        )
                    elif i == 2:
                        # Top-right corner
                        draw.polygon(
                            [(center_x, y1), (x2, y1), (x2, center_y), (center_x, center_y)],
                            fill=(rgb[0], rgb[1], rgb[2], 120),
                            outline=(rgb[0], rgb[1], rgb[2], 255),
                        )
                    elif i == 3:
                        # Bottom-left corner
                        draw.polygon(
                            [(x1, center_y), (center_x, center_y), (center_x, y2), (x1, y2)],
                            fill=(rgb[0], rgb[1], rgb[2], 120),
                            outline=(rgb[0], rgb[1], rgb[2], 255),
                        )
                # Draw thick border for overlapping patches
                draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(255, 255, 0, 255), width=4)
    
    # Composite overlay on resized image
    result = Image.alpha_composite(resized_img.convert("RGBA"), overlay)
    
    # Add legend with increased height for larger boxes and text
    legend_height = 200  # Increased from 150
    legend_img = Image.new("RGBA", (img_w, img_h + legend_height), (255, 255, 255, 255))
    legend_img.paste(result, (0, 0))
    legend_draw = ImageDraw.Draw(legend_img)
    
    # Try to load a larger font if available
    try:
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                font = ImageFont.load_default()
    except:
        font = None
    
    # Get English names for regions
    region_names_en = {
        "Lung and Pleura": "Lung and Pleura",
        "Mediastinum and Airway": "Mediastinum and Airway",
        "Cardiac": "Cardiac",
        "Bones and Soft Tissue": "Bones and Soft Tissue",
        "Diaphragm and Abdomen": "Diaphragm and Abdomen"
    }
    
    # Draw legend at bottom with enlarged boxes and text
    y_start = img_h + 20
    x_start = 30
    # Enlarged color box size: 35x25 (was 20x15)
    box_w, box_h = 35, 25
    pad = 8
    x = x_start
    y = y_start
    
    # Legend for single-region patches
    if font:
        legend_draw.text((x, y), "Single region:", fill=(0, 0, 0, 255), font=font)
    else:
        legend_draw.text((x, y), "Single region:", fill=(0, 0, 0, 255))
    y += 35  # Increased spacing for larger boxes
    
    for region_key in sorted(region_patch_map.keys()):
        region_name = region_names_en.get(region_key, region_key.replace("_", " ").title())
        text_width = len(region_name) * 10  # Approximate width
        # Wrap to next line if needed
        if x + box_w + text_width + 50 > img_w:
            x = x_start
            y += 40  # Larger spacing for enlarged boxes
        color = color_map.get(region_key, "lime")
        rgb = ImageColor.getrgb(color)
        # Draw enlarged color box
        legend_draw.rectangle([x, y, x + box_w, y + box_h], fill=color, outline=(0, 0, 0, 255), width=2)
        # Draw enlarged text with English name
        if font:
            legend_draw.text((x + box_w + 8, y + 2), region_name, fill=(0, 0, 0, 255), font=font)
        else:
            legend_draw.text((x + box_w + 8, y + 2), region_name, fill=(0, 0, 0, 255))
        x += box_w + text_width + 60  # Larger spacing between items
    
    # Legend for overlapping patches
    y += 40  # Increased spacing
    x = x_start
    if font:
        legend_draw.text((x, y), "Overlapping (multiple regions):", fill=(0, 0, 0, 255), font=font)
        y += 30
        legend_draw.text((x, y), "Yellow border = patch belongs to multiple regions", fill=(255, 200, 0, 255), font=font)
    else:
        legend_draw.text((x, y), "Overlapping (multiple regions):", fill=(0, 0, 0, 255))
        y += 30
        legend_draw.text((x, y), "Yellow border = patch belongs to multiple regions", fill=(255, 200, 0, 255))
    
    legend_img.convert("RGB").save(out_path)

    
def process_single_sample(json_path: str,
                          img_path: str,
                          output_path: str,
                          all_bbox_regions_path: str = ALL_BBOX_REGIONS_PATH,
                          mi_json_path: Optional[str] = MIMIC_JSON_PATH,
                          save_intermediate: bool = False,
                          save_visualizations: bool = False,
                          vis_output_dir: Optional[str] = None,
                          regions: Optional[Dict[str, List[str]]] = None,
                          tag_dict: Optional[Dict[str, int]] = None,
                          mi_cdrc_data: Optional[Dict] = None,
                          id_to_image_path_map: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """
    Process a single sample from JSON and image to generate dataset JSON.
    
    Args:
        json_path: Path to input JSON file (e.g., "1.json")
        img_path: Path to input image file (e.g., "1.jpg")
        output_path: Path to output dataset JSON file (e.g., "1_dataset.json")
        all_bbox_regions_path: Path to all_bbox_names_and_regions.json
        mi_json_path: Path to MI_CDRC.json (optional)
        save_intermediate: Whether to save intermediate grouped JSON
        save_visualizations: Whether to save visualization images
        vis_output_dir: Directory to save visualization images (default: same as output_path directory)
        regions: Pre-loaded regions mapping (optional, for performance)
        tag_dict: Pre-loaded tag dictionary (optional, for performance)
        mi_cdrc_data: Pre-loaded MIMIC JSON data (optional, for performance)
        id_to_image_path_map: Pre-loaded image path mapping (optional, for performance)
    
    Returns:
        Dictionary with processing statistics
    """
    # 1) Load bbox names and regions (if not provided)
    if regions is None:
        all_bbox_names, regions = load_all_bbox_and_regions(all_bbox_regions_path)
    else:
        all_bbox_names = set()  # Not used in this function, but kept for compatibility
    
    # 2) Load sample JSON
    with open(json_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)
    
    image_id = sample_data.get("image_id", "")
    
    # 3) Group structures by regions
    grouped = group_structures(sample_data, regions) #{分区名：[解剖部位1的信息，解剖部位2的信息]}
    
    # 4) Load and process image
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    # 使用与 Qwen2.5-VL 图像处理器完全相同的参数
    # patch_size=14, merge_size=2, factor=28 (每个 merged patch = 28×28 像素)
    
    orig_width, orig_height = img.size
    # 使用与 Qwen2.5-VL 图像处理器完全相同的参数进行图像缩放
    # factor=28 (patch_size * merge_size), min_pixels=3136, max_pixels=1003520
    resized_height, resized_width = smart_resize(
        orig_height,
        orig_width,
        factor=FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    
    # 实际缩放图像：调用 transformers 的 resize，与 Qwen2VLImageProcessor 保持一致（BICUBIC）
    img_np = np.array(img)
    resized_img_np = hf_resize(
        img_np,
        size=(resized_height, resized_width),
        resample=Image.Resampling.BICUBIC,
        input_data_format=None,
    )
    # 转回 PIL.Image 以便后续可视化与 bbox 绘制
    resized_img = Image.fromarray(resized_img_np.astype("uint8"))
    
    # 5) Scale bounding boxes to resized dimensions (direct stretch)
    scale_bboxes_for_resized(grouped, orig_w, orig_h, resized_width, resized_height)
    
    # 6) Compute region patch occupancy
    # 使用与 Qwen2.5-VL 图像处理器相同的 patch_side (28 = patch_size * merge_size)
    merge_side = PATCH_SIZE * MERGE_SIZE  # 28
    region_patch_map = compute_region_patch_occupancy(
        grouped, resized_width, resized_height, patch_side=merge_side, threshold_ratio=THRESHOLD_RATIO
    )
    
    # 7) Store patches in grouped structures
    for region_key, reg in grouped.items():
        patches = region_patch_map.get(region_key, [])
        reg["patches"] = patches
    
    # 7.1) Optional: Save visualizations
    if save_visualizations:
        # Determine output directory for visualizations
        if vis_output_dir is None:
            vis_output_dir = os.path.dirname(output_path)
        os.makedirs(vis_output_dir, exist_ok=True)
        
        # Get base name for output files
        base_name = os.path.splitext(os.path.basename(output_path))[0].replace("_dataset", "")
        
        # Color map for regions
        color_map = {
            "Lung and Pleura": "red",
            "Mediastinum and Airway": "orange",
            "Cardiac": "magenta",
            "Bones and Soft Tissue": "cyan",
            "Diaphragm and Abdomen": "yellow",
        }
        
        # 1. Original image with original bounding boxes
        orig_vis = draw_boxes(img, grouped, use_original=True, color_map=color_map)
        orig_vis_path = os.path.join(vis_output_dir, f"{base_name}_original_with_boxes.jpg")
        orig_vis.save(orig_vis_path)
        print(f"  Saved original image with boxes -> {orig_vis_path}")
        
        # 2. Resized image with resized bounding boxes
        # resized_img is already created above using transforms.Resize
        resized_vis = draw_boxes(resized_img, grouped, use_original=False, color_map=color_map)
        resized_vis_path = os.path.join(vis_output_dir, f"{base_name}_resized_with_boxes.jpg")
        resized_vis.save(resized_vis_path)
        print(f"  Saved resized image with boxes -> {resized_vis_path}")
        
        # 3. Region patches visualization
        merge_grid_w = math.ceil(resized_width / merge_side)
        patches_vis_path = os.path.join(vis_output_dir, f"{base_name}_region_patches.png")
        draw_region_patches_vis(resized_img, merge_side, merge_grid_w, region_patch_map, color_map, patches_vis_path)
        print(f"  Saved region patches visualization -> {patches_vis_path}")
    
    # 8) Generate messages_interleave_ra
    messages_interleave_ra = generate_messages_interleave_ra(
        grouped, image_id, mi_json_path, 
        tag_dict=tag_dict, mi_cdrc_data=mi_cdrc_data
    )

    
    # 9) Build output JSON
    # Get image_path from pre-loaded map or MI_CDRC.json
    image_path_value = []
    if id_to_image_path_map is not None:
        image_path_value = id_to_image_path_map.get(image_id, [])
    elif mi_json_path and os.path.exists(mi_json_path):
        try:
            if mi_cdrc_data is None:
                with open(mi_json_path, "r", encoding="utf-8") as f:
                    mi_cdrc_data = json.load(f)
            # Search in train, va, and test splits
            for split_key in ["train"]:
                if split_key not in mi_cdrc_data:
                    continue
                for record in mi_cdrc_data[split_key]:
                    if record.get("id") == image_id:
                        image_path_value = record.get("image_path", [])
                        break
                if image_path_value:
                    break
        except Exception:
            pass
    
    output_json = {
        "id": image_id,
        "images": image_path_value,
        "messages_interleave_ra": messages_interleave_ra
    }
    
    # 10) Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    
    # Optional: Save intermediate grouped JSON
    # if save_intermediate:
    #     intermediate_path = output_path.replace("_dataset.json", "_grouped.json")
    #     intermediate_json = {
    #         "image_id": image_id,
    #         "patient_id": sample_data.get("patient_id"),
    #         "study_id": sample_data.get("study_id"),
    #         "anatomical_regions": grouped,
    #         "patch_info": {
    #             "patch_size": PATCH_SIZE,
    #             "merge_size": MERGE_SIZE,
    #             "resized_width": resized_w,
    #             "resized_height": resized_h,
    #             "merge_grid_h": math.ceil(resized_h / merge_side),
    #             "merge_grid_w": math.ceil(resized_w / merge_side),
    #             "occupied_patch_side": merge_side,
    #             "occupied_threshold_ratio": 0.5,
    #         },
    #     }
    #     with open(intermediate_path, "w", encoding="utf-8") as f:
    #         json.dump(intermediate_json, f, ensure_ascii=False, indent=2)
    
    # Return statistics
    regions_with_desc = sum(1 for reg in grouped.values() 
                            if any(s.get("descriptions") for s in reg.get("structures", [])))
    

    return {
        "image_id": image_id,
        "regions_count": len(grouped),
        "regions_with_descriptions": regions_with_desc,
        "resized_size": (resized_width, resized_height),
        "output_path": output_path
    }


def _process_single_sample_wrapper(args: Tuple) -> Optional[Dict[str, Any]]:
    """
    多进程处理的包装函数。
    接受一个元组参数，解包后处理单个样本。
    这个函数直接处理，避免不必要的文件I/O。
    """
    (json_path, img_path, image_id, image_path_value, all_bbox_regions_path,
     mi_json_path, regions, tag_dict, mi_cdrc_data, id_to_image_path_map) = args
    
    # 设置进程级别的错误处理
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        # 1. 加载样本JSON
        with open(json_path, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        
        # 2. 按区域分组结构
        grouped = group_structures(sample_data, regions)
        
        # 3. 加载和处理图像
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        resized_height, resized_width = smart_resize(
            orig_h, orig_w, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        img_np = np.array(img)
        resized_img_np = hf_resize(
            img_np, size=(resized_height, resized_width),
            resample=Image.Resampling.BICUBIC, input_data_format=None
        )
        resized_img = Image.fromarray(resized_img_np.astype("uint8"))
        
        # 4. 缩放边界框
        scale_bboxes_for_resized(grouped, orig_w, orig_h, resized_width, resized_height)
        
        # 5. 计算区域patch占用
        merge_side = PATCH_SIZE * MERGE_SIZE
        region_patch_map = compute_region_patch_occupancy(
            grouped, resized_width, resized_height, patch_side=merge_side, threshold_ratio=THRESHOLD_RATIO
        )
        for region_key, reg in grouped.items():
            reg["patches"] = region_patch_map.get(region_key, [])
        
        # 6. 生成消息
        messages_interleave_ra = generate_messages_interleave_ra(
            grouped, image_id, mi_json_path, tag_dict=tag_dict, mi_cdrc_data=mi_cdrc_data
        )
        
        # 7. 返回结果
        return {
            "id": image_id,
            "images": image_path_value,
            "messages_interleave_ra": messages_interleave_ra
        }
    except Exception as e:
        print(f"Error processing {image_id}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def process_scene_graph_batch(scene_graph_dir: str,
                              mimic_img_dir: str,
                              output_json_path: str,
                              all_bbox_regions_path: str = ALL_BBOX_REGIONS_PATH,
                              mi_josn_path: Optional[str] = MIMIC_JSON_PATH,
                              num_workers: Optional[int] = None) -> None:
    """
    Args:
        scene_graph_dir: Directory containing scene_graph JSON files
        mimic_img_dir: Directory containing mimic images (e.g., sample/mimic)
        output_json_path: Path to output combined JSON file
        all_bbox_regions_path: Path to all_bbox_names_and_regions.json
        mi_json_path: Path to MIMIC_annotation.json (required for fast lookup)
        num_workers: Number of parallel workers (default: cpu_count() - 2, minimum 1)
    """
    import glob
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback if tqdm is not installed
        def tqdm(iterable, desc=""):
            print(desc)
            return iterable

    # 确定工作进程数
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # 留2个核心给系统
    print(f"Using {num_workers} parallel workers")

    # 预加载共享数据
    print("Pre-loading shared data...")
    start_time = time.time()
    
    # 1. 加载 bbox regions
    print("  Loading bbox regions...")
    _, regions = load_all_bbox_and_regions(all_bbox_regions_path)
    
    # 2. 加载 tag dictionary
    print("  Loading tag dictionary...")
    tag_dict = {}
    if os.path.exists(TAG_PATH):
        with open(TAG_PATH, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                study_id = row['study_id'].strip() if row.get('study_id') else None
                if study_id:
                    if row.get('No Finding') == '1.0':
                        tag_dict[study_id] = 0
                    else:
                        tag_dict[study_id] = 1
    
    # 3. 加载 MIMIC JSON 数据
    print("  Loading MIMIC JSON data...")
    mi_cdrc_data = None
    if mi_josn_path and os.path.exists(mi_josn_path):
        with open(mi_josn_path, "r", encoding="utf-8") as f:
            mi_cdrc_data = json.load(f)
    
    # 4. 构建图像路径映射
    print("  Building image path map...")
    id_to_full_path_map, id_to_image_path_map = build_image_path_map(mi_josn_path, mimic_img_dir)
    print(f"  Image path map built: {len(id_to_full_path_map)} images found")
    
    load_time = time.time() - start_time
    print(f"Finished pre-loading in {load_time:.2f} seconds")
    print(f"  Regions loaded: {len(regions)} regions")
    print(f"  Tag dict size: {len(tag_dict)} entries")
    if mi_cdrc_data:
        print(f"  MIMIC data loaded: {len(mi_cdrc_data.get('train', []))} train records")
    
    if not id_to_full_path_map:
        print("Warning: No images found in annotation.json. Processing will skip all samples.")
        return

    # 获取所有JSON文件
    print("Scanning for JSON files...")
    if os.path.isfile(scene_graph_dir) and scene_graph_dir.endswith(".txt"):
        print(f"---- scene_graph_dir: {scene_graph_dir}")
        with open(scene_graph_dir, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            json_files = [line.strip() for line in all_lines if line.strip()]
            print(f"Loaded {len(json_files)} JSON paths from list file {scene_graph_dir}")
    else:
        json_files = glob.glob(os.path.join(scene_graph_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {scene_graph_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files in scene_graph directory")
    
    # 准备任务列表（添加进度条）
    print("Preparing task list...")
    tasks = []
    skipped_count = 0
    
    for json_path in tqdm(json_files, desc="Preparing tasks"):
        try:
            # 快速读取 image_id（不加载完整JSON）
            with open(json_path, "r", encoding="utf-8") as f:
                sample_data = json.load(f)
            image_id = sample_data.get("image_id", "")
            if not image_id:
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                image_id = base_name
            
            img_path = id_to_full_path_map.get(image_id)
            if not img_path or not os.path.exists(img_path):
                skipped_count += 1
                continue
            
            image_path_value = id_to_image_path_map.get(image_id, [])
            tasks.append((
                json_path, img_path, image_id, image_path_value,
                all_bbox_regions_path, mi_josn_path,
                regions, tag_dict, mi_cdrc_data, id_to_image_path_map
            ))
        except Exception as e:
            print(f"Error preparing task for {json_path}: {e}", flush=True)
            continue
    
    print(f"Prepared {len(tasks)} tasks (skipped {skipped_count} files without images)", flush=True)
    print(f"Task list size: {len(tasks)} items", flush=True)
    
    # 可选：测试模式（只处理前N个任务，用于调试）
    TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"
    TEST_LIMIT = int(os.environ.get("TEST_LIMIT", "10"))
    if TEST_MODE:
        print(f"  TEST MODE: Only processing first {TEST_LIMIT} tasks", flush=True)
        tasks = tasks[:TEST_LIMIT]
    
    # 估算共享数据大小（用于调试和优化）
    import sys
    total_shared_size = 0
    if mi_cdrc_data:
        # 更准确的估算：使用pickle大小
        import pickle
        mi_cdrc_size = len(pickle.dumps(mi_cdrc_data)) / (1024 * 1024)  # MB
        total_shared_size += mi_cdrc_size
        print(f"  MIMIC JSON data size: ~{mi_cdrc_size:.2f} MB")
    if tag_dict:
        tag_dict_size = len(pickle.dumps(tag_dict)) / (1024 * 1024)  # MB
        total_shared_size += tag_dict_size
        print(f"  Tag dictionary size: ~{tag_dict_size:.2f} MB")
    if regions:
        regions_size = len(pickle.dumps(regions)) / (1024 * 1024)  # MB
        total_shared_size += regions_size
        print(f"  Regions data size: ~{regions_size:.2f} MB")
    
    # 估算总序列化大小（每个进程都会复制一份）
    estimated_total = total_shared_size * num_workers / (1024)  # GB
    print(f"  Estimated total memory for shared data: ~{estimated_total:.2f} GB (across {num_workers} workers)")
    if estimated_total > 50:  # 如果超过50GB
        print(f"  WARNING: Large shared data size may cause slow process pool initialization!")
        print(f"  Consider reducing num_workers or optimizing data structures.")
    
    # 使用多进程池处理
    all_results = []
    processed_count = 0
    error_count = 0
    
    print(f"Starting parallel processing with {num_workers} workers...")
    print("  Initializing process pool (this may take a moment for large datasets)...", flush=True)
    print("  Note: Large shared data will be serialized to each worker process.", flush=True)
    print("  This may take several minutes if data is very large.", flush=True)
    pool_init_start = time.time()
    start_time = time.time()
    
    # 在后台线程中监控初始化进度
    import threading
    init_warning_printed = threading.Event()
    
    def check_init_progress():
        """每30秒检查一次初始化进度"""
        time.sleep(30)
        if not init_warning_printed.is_set():
            elapsed = time.time() - pool_init_start
            print(f"  Still initializing... ({elapsed:.1f}s elapsed)", flush=True)
            print(f"  This is normal for large datasets. Please wait...", flush=True)
            init_warning_printed.set()
    
    progress_thread = threading.Thread(target=check_init_progress, daemon=True)
    progress_thread.start()
    
    try:
        # 使用 context manager 确保进程池正确关闭
        pool = Pool(processes=num_workers)
        pool_init_time = time.time() - pool_init_start
        init_warning_printed.set()  # 停止警告
        print(f"  Process pool initialized in {pool_init_time:.2f} seconds!", flush=True)
        print("  Starting to process samples...", flush=True)
        
        # 使用 imap_unordered 替代 imap，更不容易死锁
        # 虽然顺序可能不同，但对于独立任务来说不影响结果
        try:
            results = list(tqdm(
                pool.imap_unordered(_process_single_sample_wrapper, tasks, chunksize=10),
                total=len(tasks),
                desc="Processing samples",
                mininterval=1.0  # 至少每秒更新一次进度条
            ))
        finally:
            # 确保进程池正确关闭
            print("  Closing process pool...", flush=True)
            pool.close()
            pool.join()
            print("  Process pool closed.", flush=True)
            
    except KeyboardInterrupt:
        print("\n  Interrupted by user, closing pool...", flush=True)
        if 'pool' in locals():
            pool.terminate()
            pool.join()
        raise
    except Exception as e:
        print(f"Error in process pool: {e}", flush=True)
        import traceback
        traceback.print_exc()
        if 'pool' in locals():
            print("  Terminating pool due to error...", flush=True)
            pool.terminate()
            pool.join()
        raise
    
    # 收集结果
    for result in results:
        if result is not None:
            all_results.append(result)
            processed_count += 1
        else:
            error_count += 1
    
    process_time = time.time() - start_time
    print(f"Processing completed in {process_time:.2f} seconds")
    print(f"Average time per sample: {process_time/len(tasks):.3f} seconds")
    
    # 保存结果
    print("Saving results...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nBatch processing complete:")
    print(f"  - Total JSON files: {len(json_files)}")
    print(f"  - Processed (with images): {processed_count}")
    print(f"  - Skipped (no image): {skipped_count}")
    print(f"  - Errors: {error_count}")
    print(f"  - Output: {output_json_path}")
    print(f"  - Total time: {load_time + process_time:.2f} seconds")







if __name__ == "__main__":
    import os
    
    print("Starting batch processing...")
    print(f"  Scene graph directory: {SCENE_GRAPH_DIR}")
    print(f"  Mimic image directory: {MIMIC_IMG_DIR}")
    print(f"  Output JSON: {OUTPUT_JSON_PATH}")
    
    # 从环境变量读取工作进程数，如果没有设置则使用默认值
    num_workers = None
    

    if "NUM_WORKERS" in os.environ:
        try:
            num_workers = int(os.environ["NUM_WORKERS"])
            print(f"  Using {num_workers} workers from environment variable")
        except ValueError:
            print(f"  Warning: Invalid NUM_WORKERS value, using default")
    else:
        print(f"  Using default number of workers (CPU count - 2)")
    
    print("MAX_PIXELS:", MAX_PIXELS)
    print()
    
    process_scene_graph_batch(
        scene_graph_dir=SCENE_GRAPH_DIR,
        mimic_img_dir=MIMIC_IMG_DIR,
        output_json_path=OUTPUT_JSON_PATH,
        num_workers=num_workers
    )

