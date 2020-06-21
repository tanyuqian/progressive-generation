import pickle
from glob import glob
import fire
from prettytable import PrettyTable


def main(dataset, metric):
    log_dir = f'eval_logs/{metric}'

    results = {}
    keys = None
    for filename in glob(f'{log_dir}/{dataset}_*.pickle'):
        label = filename.split('/')[-1][:-len('.pickle')]
        results[label] = pickle.load(open(filename, 'rb'))
        keys = results[label].keys()

    labels = sorted(results.keys(), key=lambda s: s[::-1])

    table = PrettyTable([' '] + labels)
    for key in keys:
        new_raw = [key]
        for label in labels:
            if isinstance(results[label][key], float):
                if metric == 'tfidf_distance':
                    new_raw.append(f'{results[label][key] * 1000:.4f}')
                else:
                    new_raw.append(f'{results[label][key]:.4f}')
            else:
                new_raw.append(results[label][key])
        table.add_row(new_raw)

    print(table)


if __name__ == '__main__':
    fire.Fire(main)
