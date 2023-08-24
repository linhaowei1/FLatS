from logging import shutdown
import os
import random 

seeds = [13, 21, 42, 87, 100]
shot  = 16
num_labels = 12
kshot_dir = '{}-shot'.format(shot)

if not os.path.exists(kshot_dir):
    os.makedirs(kshot_dir)

for seed in seeds:
    output_dir = '{}/{}-{}'.format(kshot_dir, shot, seed)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    random.seed(seed)
    os.system('cp test.tsv {}/test.tsv'.format(output_dir))
    samples = [[] for _ in range(num_labels)]
    labels_map = {"weather/find": 0, "weather/checkSunrise": 1, "weather/checkSunset": 2,
                    "alarm/snooze_alarm": 3, "alarm/set_alarm": 4, "alarm/cancel_alarm": 5, "alarm/time_left_on_alarm": 6,
                    "alarm/show_alarms": 7, "alarm/modify_alarm": 8,
                    "reminder/set_reminder": 9, "reminder/cancel_reminder": 10, "reminder/show_reminders": 11,
                    "outOfDomain": -1}
    with open('OODRemovedtrain.tsv', 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            category, _, text, _ = line.split('\t')
            label = labels_map[category]
            if label != -1:
                samples[label].append(line)
    with open('{}/train.tsv'.format(output_dir), 'w') as f:
        for label in range(num_labels):
            for line in samples[label][:shot]:
                f.write(line)
    with open('{}/dev.tsv'.format(output_dir), 'w') as f:
        for label in range(num_labels):
            for line in samples[label][shot:2*shot]:
                f.write(line)
    

