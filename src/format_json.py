import json

with open('datasets/L2_ARCTIC/ori_fa_segmentation.json') as f:
    d = json.load(f)

formatted_d = {}

for key, segments in d.items():
    formatted_segments = []
    for start_frame, end_frame in segments:
        formatted_segments.append([start_frame / 100, end_frame / 100])
    formatted_d[key] = formatted_segments

with open('datasets/L2_ARCTIC/fa_segmentation.json', 'w') as f:
    json.dump(formatted_d, f, indent=4)