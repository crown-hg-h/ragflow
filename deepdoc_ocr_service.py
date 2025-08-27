#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于deepDoc的OCR服务
实现与PaddleOCR相同的API接口
"""

import logging
import sys
import os
from flask import Flask, request, jsonify
import numpy as np
import cv2
# 移除 fitz 依赖，使用其他方式处理PDF

# 添加项目路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from deepdoc.vision.ocr import OCR
    from deepdoc.parser.pdf_parser import PlainParser
except ImportError as e:
    print(f"导入deepdoc模块失败: {e}")
    print("请确保您在RAGFlow项目目录中运行此脚本")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 初始化OCR实例
try:
    ocr = OCR()
    logger.info("deepDoc OCR初始化成功")
except Exception as e:
    logger.error(f"OCR初始化失败: {e}")
    ocr = None

# 初始化PDF解析器
try:
    pdf_parser = PlainParser()
    logger.info("PDF解析器初始化成功")
except Exception as e:
    logger.error(f"PDF解析器初始化失败: {e}")
    pdf_parser = None


# 移除基于fitz的PDF OCR处理函数
# 现在直接使用PDF文本解析器处理PDF文件


def process_pdf_with_parser(file_content):
    """使用PDF解析器处理PDF文件"""
    try:
        if pdf_parser:
            # 使用deepDoc的PlainParser
            parsed_result, _ = pdf_parser(file_content)
            results = []
            for line, _ in parsed_result:
                if line.strip():
                    results.append(line.strip())
            return results
        else:
            raise Exception("PDF解析器未初始化")
    except Exception as e:
        logger.error(f"PDF解析失败: {e}")
        raise e


@app.route("/ocr_pdf", methods=["POST"])
def process_pdf():
    """处理PDF文件的OCR"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        file_content = file.read()
        if len(file_content) == 0:
            return jsonify({"error": "Empty file"}), 400

        # 使用PDF文本解析器处理PDF文件
        results = process_pdf_with_parser(file_content)
        logger.info(f"PDF文本解析完成，共{len(results)}条记录")

        return jsonify({"doc_list": results})

    except Exception as e:
        logger.error(f"PDF处理错误: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ocr_img", methods=["POST"])
def process_image():
    """处理图片文件的OCR"""
    try:
        if "imgs" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["imgs"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        img_bytes = file.read()
        if len(img_bytes) == 0:
            return jsonify({"error": "Empty file"}), 400

        if not ocr:
            return jsonify({"error": "OCR服务未初始化"}), 500

        # 解码图片
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # 使用deepDoc OCR处理
        ocr_result = ocr(img, device_id=0, cls=True)
        logger.info(f"图片OCR结果: {len(ocr_result) if ocr_result else 0}个文本块")

        text = ""
        if ocr_result:
            texts = []
            for box_info in ocr_result:
                if len(box_info) >= 2:
                    bbox, (extracted_text, score) = box_info
                    if score >= 0.5:  # 过滤低置信度文本
                        texts.append(extracted_text)
            text = " ".join(texts)

        logger.info(f"提取的文本: {text[:100]}..." if len(text) > 100 else f"提取的文本: {text}")
        return jsonify({"text": text})

    except Exception as e:
        logger.error(f"图片处理错误: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ocr_img", methods=["GET"])
def get_image_service_status():
    """获取图片OCR服务状态"""
    return jsonify(
        {
            "service": "deepDoc OCR Image Processing",
            "status": "running" if ocr else "error",
            "methods": ["POST"],
            "description": "Send POST request with 'imgs' file parameter to process images",
            "engine": "deepDoc OCR",
        }
    )


@app.route("/ocr_pdf", methods=["GET"])
def get_pdf_service_status():
    """获取PDF服务状态"""
    return jsonify(
        {
            "service": "deepDoc PDF Text Processing",
            "status": "running" if pdf_parser else "error",
            "methods": ["POST"],
            "description": "Send POST request with 'file' parameter to process PDF documents",
            "engine": "deepDoc PDF Parser",
        }
    )


@app.route("/", methods=["GET"])
def health_check():
    """健康检查"""
    return jsonify(
        {
            "service": "Smart Project Review OCR (deepDoc)",
            "status": "running",
            "ocr_status": "ready" if ocr else "error",
            "pdf_parser_status": "ready" if pdf_parser else "error",
            "endpoints": {"/ocr_img": "Image OCR processing", "/ocr_pdf": "PDF text extraction"},
            "engine": "RAGFlow deepDoc",
        }
    )


@app.route("/health", methods=["GET"])
def detailed_health():
    """详细的健康检查"""
    return jsonify(
        {
            "service": "deepDoc OCR Service",
            "status": "healthy",
            "components": {
                "ocr": {"status": "ready" if ocr else "error", "description": "deepDoc OCR引擎"},
                "pdf_parser": {"status": "ready" if pdf_parser else "error", "description": "deepDoc PDF解析器"},
            },
            "version": "1.0.0",
            "engine": "RAGFlow deepDoc",
        }
    )


if __name__ == "__main__":
    # 检查依赖
    if not ocr and not pdf_parser:
        logger.error("OCR和PDF解析器都未初始化成功，服务无法正常运行")
        sys.exit(1)
    elif not ocr:
        logger.warning("OCR未初始化，只能使用PDF文本解析功能")
    elif not pdf_parser:
        logger.warning("PDF解析器未初始化，无法处理PDF文件")

    logger.info("启动deepDoc OCR服务...")
    logger.info("服务地址: http://0.0.0.0:12306")
    logger.info("接口:")
    logger.info("  POST /ocr_img - 图片OCR")
    logger.info("  POST /ocr_pdf - PDF文本提取")
    logger.info("  GET  /        - 健康检查")
    logger.info("  GET  /health  - 详细健康检查")

    app.run(host="0.0.0.0", port=12306, debug=True)
