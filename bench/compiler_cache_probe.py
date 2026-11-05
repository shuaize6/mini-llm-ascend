"""在两个独立进程中运行同一命令，核对算子磁盘缓存和首次执行时间。"""
import argparse
import hashlib
import os
import time
from pathlib import Path
from bench.common import open_device, synchronize, write_report


def snapshot(root):
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size,
             'mtime_ns': p.stat().st_mtime_ns, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.cache_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    os.environ['ACL_OP_COMPILER_CACHE_MODE'] = 'enable'
    os.environ['ACL_OP_COMPILER_CACHE_DIR'] = str(root)
    report = {'pid':os.getpid(),'cache_dir':str(root),'before':snapshot(root),'status':'running'}
    try:
        torch,device,dtype = open_device('npu','float16')
        torch.npu.set_option({'ACL_OP_COMPILER_CACHE_MODE':'enable','ACL_OP_COMPILER_CACHE_DIR':str(root)})
        a = torch.ones((1,17,128),device=device,dtype=dtype)
        b = torch.ones((1,128,19),device=device,dtype=dtype)
        synchronize(torch,device)
        report['times_s'] = []
        for _ in range(2):
            start=time.perf_counter()
            output=torch.matmul(a,b)
            synchronize(torch,device)
            report['times_s'].append(time.perf_counter()-start)
        report['correct'] = bool((output.cpu()==128).all().item())
        report['status'] = 'ok' if report['correct'] else 'error'
        report['after'] = snapshot(root)
    except Exception as exc:
        report.update(status='error',error=repr(exc))
    write_report(args.output,report)
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':
    raise SystemExit(main())
