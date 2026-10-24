"""Analyze existing external measurements; performs no device I/O benchmark.

Usage: python analyze.py --vlm-pdf /path/to/2511.18692v1.pdf
Dependencies: numpy, matplotlib, PyMuPDF.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)


def save_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spread(values):
    v = np.asarray(values, dtype=float)
    return dict(n=len(v), mean=float(v.mean()), min=float(v.min()),
                max=float(v.max()), peak_to_peak_pct_of_mean=float(100*np.ptp(v)/v.mean()))


def nvme_data():
    base = ROOT / 'raw' / 'NVMeBenchmarks'
    manifest = json.loads((base / 'manifest.json').read_text())
    rows = []
    for item in manifest['files']:
        raw = (base / item['path']).read_bytes()
        # apply_patch can normalize the final newline; verify Git blob identity
        # against either representation, without changing JSON numeric values.
        candidates = [raw, raw.rstrip(b'\n'), raw.rstrip(b'\n')+b'\n']
        assert any(hashlib.sha1(b'blob '+str(len(x)).encode()+b'\0'+x).hexdigest()
                   == item['sha'] for x in candidates), item['path']
        d = json.loads(raw)
        assert len(d['jobs']) == 1
        j = d['jobs'][0]
        g, o, r = d['global options'], j['job options'], j['read']
        assert j['error'] == 0 and g['rw'] == 'read' and g['direct'] == '1'
        assert g['ioengine'] == 'io_uring' and o['numjobs'] == '1'
        bs, qd, bw = int(o['bs']), int(o['iodepth']), r['bw_bytes']
        # Published byte and completion counters differ slightly (up to 0.0015%).
        # Check unit consistency at 0.01%, retaining the discrepancy for audit.
        counter_relative_difference = bw / (r['iops'] * bs) - 1
        assert abs(counter_relative_difference) < 1e-4
        rows.append(dict(block_bytes=bs, block_kib=bs/1024, queue_depth=qd,
                         numjobs=1, bandwidth_bytes_s=bw, bandwidth_mib_s=bw/2**20,
                         amortized_us=1e6*bs/bw,
                         amortized_us_per_kib=1e6*1024/bw,
                         reported_mean_lat_us=r['lat_ns']['mean']/1000,
                         reported_mean_clat_us=r['clat_ns']['mean']/1000,
                         bandwidth_iops_relative_difference=counter_relative_difference,
                         runtime_ms=r['runtime'], source_path=item['path'],
                         source_url='https://github.com/stonet-research/NVMeBenchmarks/blob/'
                                    +manifest['commit']+'/'+item['path']))
    rows.sort(key=lambda r: (r['queue_depth'], r['block_bytes']))
    assert len(rows) == 18
    save_csv('zns_raw_read_summary.csv', rows)
    return rows


def intel_data():
    # Published rounded values: ADMS 2010, Tables 8 and 10, SINGLE SSD READ.
    # The throughput and latency columns come from DIFFERENT QD workloads.
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    bw = [196, 239, 248, 248, 252, 252, 252, 252]
    lat = [.064, .093, .152, .282, .529, 1.062, 2.115, 4.249]
    rows = [dict(block_kb_as_published=s, sequential_read_mb_s_qd32=b,
                 sequential_read_latency_ms_qd1=t, latency_us_per_published_kb=t*1000/s,
                 provenance='published_table_transcription_not_raw_repeats')
            for s, b, t in zip(sizes, bw, lat)]
    save_csv('intel_x25e_published_tables.csv', rows)
    return rows


def vlm_data(pdf):
    doc = pymupdf.open(pdf)
    page = doc[3]
    assert 'Figure 4' in page.get_text()
    ds = page.get_drawings()
    # Tick/grid coordinates, NOT text bounding-box centers.
    x0, x500 = ds[4]['rect'].x0, ds[14]['rect'].x0
    y0, y6 = ds[16]['rect'].y0, ds[22]['rect'].y0
    assert abs((x500-x0)-129.083) < .01
    rows = []
    # Circle centers recover all published mean markers, even where matplotlib
    # simplified the connecting polyline. Excludes legend markers and error bands.
    for device, start, stop in [('Nano', 25, 536), ('AGX', 539, 794)]:
        for index in range(start, stop):
            d = ds[index]
            assert d['type'] == 'fs' and len(d['items']) == 8
            rect = d['rect']
            x, y = (rect.x0+rect.x1)/2, (rect.y0+rect.y1)/2
            length = (x-x0)*500/(x500-x0)
            assert abs(length-round(length)) < .01
            b = (y0-y)*6/(y0-y6)
            rows.append(dict(device=device, block_kb_axis=round(length),
                             throughput_gb_s_axis=b, inverse_throughput=1/b,
                             provenance='PDF_vector_marker_digitization_not_raw_repeats'))
    assert len(rows) == 766
    save_csv('vlmflash_fig4a_digitized.csv', rows)
    page.get_pixmap(matrix=pymupdf.Matrix(3, 3),
                    clip=pymupdf.Rect(133, 71, 301, 179)).save(OUT / 'vlmflash_fig4a_source.png')
    meta = dict(source='https://arxiv.org/pdf/2511.18692v1#page=4',
                pdf_sha256=hashlib.sha256(Path(pdf).read_bytes()).hexdigest(),
                page_index=3, x0=x0, x500=x500, y0=y0, y6=y6,
                drawing_marker_ranges_inclusive={'Nano':[25,535], 'AGX':[539,793]},
                note='Published axis units retained; no assumed decimal/binary conversion.')
    (OUT / 'vlm_digitization_metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm-pdf', required=True)
    args = parser.parse_args()
    nv, intel, vlm = nvme_data(), intel_data(), vlm_data(args.vlm_pdf)
    metrics = {'vlm_tail_350_to_500_published_kb': {},
               'zns_tail_64_to_128_kib': {},
               'intel_qd1_tail_128_to_1024_published_kb': {}}
    for device in ['AGX', 'Nano']:
        tail = [r for r in vlm if r['device']==device and 350 <= r['block_kb_axis'] <= 500]
        metrics['vlm_tail_350_to_500_published_kb'][device] = {
            'throughput_axis_gb_s': spread([r['throughput_gb_s_axis'] for r in tail]),
            'inverse_throughput': spread([r['inverse_throughput'] for r in tail])}
    for qd in [1, 32, 64]:
        tail = [r for r in nv if r['queue_depth']==qd and r['block_kib']>=64]
        b = [r['bandwidth_mib_s'] for r in tail]
        metrics['zns_tail_64_to_128_kib'][str(qd)] = {
            'throughput_mib_s': spread(b), 'change_64_to_128_pct': 100*(b[1]/b[0]-1)}
    tail = [r for r in intel if r['block_kb_as_published']>=128]
    metrics['intel_qd1_tail_128_to_1024_published_kb'] = spread(
        [r['latency_us_per_published_kb'] for r in tail])
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), layout='constrained')
    colors = {'AGX':'#009d9a', 'Nano':'#d02671'}
    for device in ['AGX', 'Nano']:
        r = [r for r in vlm if r['device']==device]
        x = np.array([a['block_kb_axis'] for a in r])
        bw = np.array([a['throughput_gb_s_axis'] for a in r])
        axes[0,0].plot(x, bw, label=device, c=colors[device], lw=1)
        sel = (x>=350)&(x<=500)
        axes[0,1].plot(x[sel], (1/bw[sel])/np.mean(1/bw[sel]),
                       c=colors[device], label=device, lw=1)
    axes[0,0].set(title='VLMFlash Fig. 4(a): digitized figure',
                  xlabel='Block size (published KB)', ylabel='Throughput (published GB/s)')
    axes[0,0].axvspan(350, 500, alpha=.10, color='grey')
    axes[0,1].set(title='VLMFlash: cost per byte in 350–500 KB band',
                  xlabel='Block size (published KB)', ylabel='Inverse throughput / band mean')
    axes[0,1].axhline(1, c='grey', lw=.8, ls='--')
    for qd in [1,32,64]:
        r = [r for r in nv if r['queue_depth']==qd]
        axes[1,0].plot([a['block_kib'] for a in r], [a['bandwidth_mib_s'] for a in r],
                       '-o', markersize=4, label=f'QD {qd}')
    axes[1,0].set(title='ZN540: original fio JSON, one job',
                  xlabel='Block size (KiB)', ylabel='Read throughput (MiB/s)')
    axes[1,0].set_xscale('log', base=2)
    axes[1,0].set_xticks([4,8,16,32,64,128], ['4','8','16','32','64','128'])
    axes[1,1].plot([r['block_kb_as_published'] for r in intel],
                   [r['latency_us_per_published_kb'] for r in intel], '-o', label='QD 1')
    axes[1,1].set(title='Intel X25-E: published latency table',
                  xlabel='Block size (published KB)', ylabel='Measured latency / size (µs/KB)')
    axes[1,1].set_xscale('log', base=2)
    axes[1,1].set_xticks([8,32,128,512,1024], ['8','32','128','512','1024'])
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(OUT / 'external_saturation_evidence.png', dpi=180)
    fig.savefig(OUT / 'external_saturation_evidence.pdf')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
