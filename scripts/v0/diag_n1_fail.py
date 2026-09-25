"""诊断：深 20q 电路在并行池中的失败原因。"""
import sys, json
sys.path.insert(0, 'src')
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from run_n1_calib_diag_helper import _worker


def main():
    manifest = json.load(open('traindata/v0/manifest.json'))
    deep20 = [x['file'] for x in manifest if x['nq'] == 20 and x['n2q'] >= 60][:6]
    tasks = []
    for c in deep20:
        for variant, n_pert in [("hop", 0), ("noise", 0), ("hop", 3),
                                ("noise", 3), ("hop", 8)]:
            tasks.append((f'traindata/v0/{c}',
                          'traindata/topo/tianyan287_20q.json',
                          variant, n_pert, 100, 64))
    reasons = Counter()
    with ProcessPoolExecutor(max_workers=15) as pool:
        futs = [pool.submit(_worker, t) for t in tasks]
        for f in as_completed(futs):
            r = f.result()
            reasons[r.get('reason', 'OK')[:80]] += 1
    for reason, n in reasons.most_common(6):
        print(n, '×', reason, flush=True)


if __name__ == "__main__":
    main()
