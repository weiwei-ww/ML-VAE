import json
from pathlib import Path

file_path = Path('datasets/SynAudioMNIST/external_data/dnn_hmm.filt.prf')
output_path = file_path.parent / 'dnn_hmm_test.json'

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
    j = 0
    for gt_phn, pred_phn in zip(*sample_lines[1:3]):
        # print(gt_phn, pred_phn, start_time, end_time)
        if '*' in gt_phn:  # ignore insertions
            continue
        elif '*' in pred_phn:  # deletion
            if len(parsed_result) > 0:
                start_time, end_time, _ = parsed_result[-1]
            else:
                start_time = end_time = 0
        else:
            start_time, end_time = sample_lines[-2][j], sample_lines[-1][j]
            j += 1

        start_time = float(start_time)
        end_time = float(end_time)
        pred_phn = pred_phn.lower()
        parsed_result.append([start_time, end_time, pred_phn])

    parsed_results[utt_id] = parsed_result

# write to json file
with open(output_path, 'w') as f:
    json.dump(parsed_results, f, indent=4)


