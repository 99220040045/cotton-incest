from flask import Flask, request, render_template, send_file, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)
model = YOLO('sigmoid.pt')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    try:
        if request.method == 'POST':
            if 'image' not in request.files:
                return jsonify({'error': 'No image uploaded'}), 400
            file = request.files['image']
            img = Image.open(file.stream)
            # Resize image to max 640x640 to reduce memory usage
            max_size = (640, 640)
            img = img.convert("RGB")
            img.thumbnail(max_size, Image.LANCZOS)
            try:
                results = model(img)
            except RuntimeError as re:
                if "out of memory" in str(re).lower():
                    return jsonify({'error': 'Server out of memory. Try a smaller image or contact admin.'}), 500
                return jsonify({'error': f'Model inference error: {str(re)}'}), 500
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
            names = results[0].names if hasattr(results[0], 'names') else model.names
            classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
            confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
            import numpy as np
            def iou(box1, box2):
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union_area = area1 + area2 - inter_area
                return inter_area / union_area if union_area > 0 else 0
            threshold = 0.3
            filtered = [(box, cls, conf) for box, cls, conf in zip(boxes, classes, confs) if conf >= threshold]
            filtered.sort(key=lambda x: x[2], reverse=True)
            merged = []
            for box, cls, conf in filtered:
                keep = True
                for mbox, mcls, mconf in merged:
                    if iou(box, mbox) > 0.5:
                        keep = False
                        break
                if keep:
                    merged.append((box, cls, conf))
            counts = {}
            output_img = img.copy()
            if len(merged) > 0:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(output_img)
                for box, cls, conf in merged:
                    name = names[int(cls)]
                    counts[name] = counts.get(name, 0) + 1
                    draw.rectangle(box.tolist(), outline='red', width=3)
                    draw.text((box[0], box[1]), f"{name} ({conf:.2f})", fill='red')
            img_io = io.BytesIO()
            if output_img.mode == 'RGBA':
                output_img = output_img.convert('RGB')
            output_img.save(img_io, 'JPEG')
            img_io.seek(0)
            import base64
            import json
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
            info_path = os.path.join(os.path.dirname(__file__), 'species_info.json')
            with open(info_path, 'r', encoding='utf-8') as f:
                species_info = json.load(f)
            details = {name: species_info.get(name, {}) for name in counts.keys()}
            warning = None
            if not counts:
                warning = "No insects detected or model is unsure. Try another image or check image quality."
            return jsonify({
                'count': counts,
                'names': list(counts.keys()),
                'details': details,
                'image': img_base64,
                'warning': warning
            }), 200
        return render_template('index.html')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
