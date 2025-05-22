import json
import os

# read json files frm a given path

file_path = './outputs_ultimate_group_recursive_search_greedy/super_resolution_x6/mpgd_wo_proj/20250520-214148_ddim100_eta0.5_scale4.0_group_recursive_n_2_temp_0.05_resample_rate_8_adaface_search/metrics_json/'

metrics = {}
for file in os.listdir(file_path):
    if file.endswith('.json'):

        # if files does not start with 'img' then skip
        if not file.startswith('img'):
            print(f'skip {file}')
            continue
        else:
            with open(os.path.join(file_path, file), 'r') as f:
                data = json.load(f)

                for key, value in data.items():
                    print('key:', key)
                    print('value:', value)
                    if float(value) <= 50:  # write only values < 50, else skip
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(float(value))            

# print('metrics:', metrics)

mean = {}
std = {}
median = {}

# calculate the average of each key
for key, value in metrics.items():
    mean[key] = sum(value) / len(value)
    median[key] = sorted(value)[len(value) // 2]
    std[key] = (sum((x - mean[key]) ** 2 for x in value) / len(value)) ** 0.5

# write the average to a json file
with open(os.path.join(file_path, 'mean_metrics.json'), 'w') as f:
    json.dump(mean, f, indent=4)

# write the median to a json file
with open(os.path.join(file_path, 'median_metrics.json'), 'w') as f:
    json.dump(median, f, indent=4)

with open(os.path.join(file_path, 'std_metrics.json'), 'w') as f:
    json.dump(std, f, indent=4)

print('done')