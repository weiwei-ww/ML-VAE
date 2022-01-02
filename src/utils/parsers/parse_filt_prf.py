import json
from pathlib import Path

file_path = Path('ctm_39phn.filt.prf')
output_path = Path('datasets/L2_ARCTIC/external_data/dnn_hmm_test.json')

if not file_path.exists():
    raise FileNotFoundError(f'file not found: {file_path.absolute()}')
if not output_path.parent.exists():
    output_path.parent.mkdir()

with open(file_path) as f:
    lines = f.readlines()  # skip the header

# skip the header
lines = lines[25:]

# only keep valid lines
valid_prefixes = ['File', 'REF:', 'HYP:', 'H_T1', 'H_T2']
lines = [line.split()[1:] for line in lines if line[:4] in valid_prefixes]
assert len(lines) % len(valid_prefixes) == 0

parsed_results = {}
num_samples = len(lines) // len(valid_prefixes)
for i in range(num_samples):
    sample_lines = lines[i * len(valid_prefixes): (i + 1) * len(valid_prefixes)]
    utt_id = sample_lines[0][0]
    parsed_result = []
    for gt_phn, pred_phn, start_time, end_time in zip(*sample_lines[1:]):
        # print(gt_phn, pred_phn, start_time, end_time)
        # ignore insertions
        if '*' in gt_phn:
            continue

        start_time = float(start_time)
        end_time = float(end_time)
        pred_phn = pred_phn.lower()
        parsed_result.append([start_time, end_time, pred_phn])
    parsed_results[utt_id] = parsed_result

# write to json file
with open(output_path, 'w') as f:
    json.dump(parsed_results, f, indent=4)


