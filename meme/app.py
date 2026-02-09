import io
import base64
import json
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, render_template, request, jsonify
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


def pil_to_dataurl(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b}"


def detect_faces(pil_image: Image.Image):
    """Detect faces using MediaPipe and return list of (x, y, w, h) rectangles."""
    # Convert PIL to RGB array
    img_array = np.array(pil_image)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Run detection
    results = face_detector.process(img_array)
    faces = []
    if results.detections:
        h, w = img_array.shape[:2]
        print(f"[人脸检测] 图片尺寸: {w}x{h}, 检测到人脸数: {len(results.detections)}")
        for i, detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)
            face_info = {'x': x, 'y': y, 'w': box_w, 'h': box_h}
            faces.append(face_info)
            face_center_x = x + box_w // 2
            print(f"  [人脸 {i+1}] 位置: ({x}, {y}), 大小: {box_w}x{box_h}, 中心X: {face_center_x}")
    else:
        print(f"[人脸检测] 未检测到人脸")
    return faces


@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        'left_b64': None,
        'right_b64': None,
        'mirrored_b64': None,
        'combined_b64': None,
        'ratio': 50,
        'direction': '左->右',
        'message': None,
        'faces': '[]',
    }

    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            context['message'] = '请上传图片文件。'
            return render_template('index.html', **context)

        try:
            image = Image.open(file.stream).convert('RGB')
            print(f"\n[图片上传] 文件名: {file.filename}, 尺寸: {image.size}")
        except Exception as e:
            context['message'] = '无法打开图片。请上传有效的 PNG/JPEG 文件。'
            print(f"[错误] 无法打开图片: {e}")
            return render_template('index.html', **context)

        ratio = float(request.form.get('ratio', 50))
        direction = request.form.get('direction', '左->右')
        w, h = image.size
        split_x = int(w * (ratio / 100.0))

        left_area = image.crop((0, 0, split_x, h))
        right_area = image.crop((split_x, 0, w, h))

        # 原图（含分割线）
        original_with_line = image.copy()
        draw0 = ImageDraw.Draw(original_with_line)
        line_width = max(1, int(h * 0.002))
        draw0.line((split_x, 0, split_x, h), fill=(255, 0, 0), width=line_width)

        # 左侧镜像 -> 右侧 (不缩放)：生成宽度为 left_width*2 的图片：左半 + 左半翻转
        left_width = split_x
        right_width = w - split_x
        if left_width > 0:
            left_mirrored = Image.new('RGB', (left_width * 2, h))
            left_mirrored.paste(left_area, (0, 0))
            flipped_left = left_area.transpose(Image.FLIP_LEFT_RIGHT)
            left_mirrored.paste(flipped_left, (left_width, 0))
        else:
            left_mirrored = None

        # 右侧镜像 -> 左侧 (不缩放)：生成宽度为 right_width*2 的图片：右半翻转 + 右半
        if right_width > 0:
            right_mirrored = Image.new('RGB', (right_width * 2, h))
            flipped_right = right_area.transpose(Image.FLIP_LEFT_RIGHT)
            right_mirrored.paste(flipped_right, (0, 0))
            right_mirrored.paste(right_area, (right_width, 0))
        else:
            right_mirrored = None

        # Prepare base64 data URLs for embedding in the page
        context['left_b64'] = pil_to_dataurl(left_area)
        context['right_b64'] = pil_to_dataurl(right_area)
        context['original_b64'] = pil_to_dataurl(original_with_line)
        context['left_mirrored_b64'] = pil_to_dataurl(left_mirrored)
        context['right_mirrored_b64'] = pil_to_dataurl(right_mirrored)
        context['ratio'] = int(ratio)
        context['direction'] = direction

    return render_template('index.html', **context)


@app.route('/detect_faces', methods=['POST'])
def detect_faces_api():
    """API endpoint for detecting faces in uploaded image."""
    file = request.files.get('image')
    if not file:
        return jsonify({'error': '没有图片上传'}), 400
    
    try:
        image = Image.open(file.stream).convert('RGB')
        print(f"\n[人脸检测API] 文件名: {file.filename}, 尺寸: {image.size}")
    except Exception as e:
        print(f"[错误] 无法打开图片: {e}")
        return jsonify({'error': '无法打开图片'}), 400
    
    # Detect faces
    faces = detect_faces(image)
    print(f"[结果] 最终返回人脸数: {len(faces)}\n")
    
    return jsonify({'faces': faces})


if __name__ == '__main__':
    app.run(debug=True)
