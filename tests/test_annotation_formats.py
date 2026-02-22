# tests/test_annotation_formats.py
"""Tests for VOC and YOLO annotation format support with polygon segmentation."""

import pytest
import numpy as np
from pathlib import Path
from src.metadata import COCOMetadataManager, mask_to_bbox, mask_to_polygon


def test_coco_metadata_manager_to_voc_xml_with_polygon():
    """Test VOC XML includes polygon when segmentation provided."""
    manager = COCOMetadataManager()
    cat_id = manager.add_category("insect")
    
    bbox = [100, 50, 100, 100]
    polygon = [[100, 50, 200, 50, 200, 150, 100, 150]]
    
    manager.add_annotation(
        image_id=1,
        category_id=cat_id,
        bbox=bbox,
        segmentation=polygon,
        area=10000,
        mask_area=10000
    )
    
    xml = manager.to_voc_xml("test.png", 640, 480, segmentation=polygon)
    
    assert "<polygon>" in xml
    assert "<x1>100</x1>" in xml
    assert "<y1>50</y1>" in xml
    assert "<x2>200</x2>" in xml
    assert "<y3>150</y3>" in xml


def test_coco_metadata_manager_to_yolo_txt_with_polygon():
    """Test YOLO TXT includes polygon when segmentation provided."""
    manager = COCOMetadataManager()
    cat_id = manager.add_category("insect")
    
    bbox = [100, 50, 100, 100]
    polygon = [[100, 50, 200, 50, 200, 150, 100, 150]]
    
    manager.add_annotation(
        image_id=1,
        category_id=cat_id,
        bbox=bbox,
        segmentation=polygon,
        area=10000,
        mask_area=10000
    )
    
    yolo_txt = manager.to_yolo_txt(width=640, height=480, segmentation=polygon)
    
    lines = yolo_txt.strip().split('\n')
    assert len(lines) == 1
    parts = lines[0].split()
    assert len(parts) == 9


def test_coco_metadata_manager_to_yolo_txt_fallback_to_bbox():
    """Test YOLO TXT falls back to bbox format when no polygon."""
    manager = COCOMetadataManager()
    cat_id = manager.add_category("insect")
    
    bbox = [100, 50, 100, 100]
    
    manager.add_annotation(
        image_id=1,
        category_id=cat_id,
        bbox=bbox,
        segmentation=None,
        area=10000,
        mask_area=10000
    )
    
    yolo_txt = manager.to_yolo_txt(width=640, height=480, segmentation=None)
    
    lines = yolo_txt.strip().split('\n')
    assert len(lines) == 1
    parts = lines[0].split()
    assert len(parts) == 5


def test_coco_metadata_manager_to_voc_xml_without_polygon():
    """Test VOC XML works without polygon (backward compatibility)."""
    manager = COCOMetadataManager()
    cat_id = manager.add_category("insect")
    
    bbox = [100, 50, 100, 100]
    
    manager.add_annotation(
        image_id=1,
        category_id=cat_id,
        bbox=bbox,
        segmentation=None,
        area=10000,
        mask_area=10000
    )
    
    xml = manager.to_voc_xml("test.png", 640, 480, segmentation=None)
    
    assert "<polygon>" not in xml
    assert "<bndbox>" in xml
    assert "<xmin>100</xmin>" in xml


def test_mask_to_polygon_empty_mask():
    """Test mask_to_polygon returns empty list for empty mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    polygon = mask_to_polygon(mask)
    assert polygon == []


def test_mask_to_polygon_single_pixel():
    """Test mask_to_polygon handles single pixel mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 50] = 255
    polygon = mask_to_polygon(mask)
    assert len(polygon) == 1
    assert len(polygon[0]) >= 2
