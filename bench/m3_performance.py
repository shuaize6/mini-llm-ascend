"""M3 同时到达双请求：串行调度 vs 批次 2，预热后交替测量。"""
import argparse
import os
import statistics
import time

from bench.common import open_device, synchronize, write_report, environment_info
from bench.m1_common import prepare_input, provenance


def run_trial(model,pool,ids,eos,torch,device,mode,tokens):
    from mini_llm_ascend.engine.scheduler import Scheduler
    scheduler=Scheduler(model,pool,max_batch_size=1 if mode=='serial' else 2)
    stamps={name:[] for name in ('a','b')}
    steps=[]
    synchronize(torch,device)
    start=time.perf_counter()
    try:
        for name in stamps:
            scheduler.submit(name,ids,tokens,eos_token_id=eos,fixed_length=True)
        while scheduler.has_pending:
            events=scheduler.step()
            now=time.perf_counter()
            steps.append([e.request_id for e in events])
            for event in events: stamps[event.request_id].append(now)
        synchronize(torch,device)
        end=time.perf_counter()
        requests={name:{'output_ids':r.output_ids,
                       'ttft_ms':1000*(stamps[name][0]-start),
                       'tpot_ms':1000*(stamps[name][-1]-stamps[name][0])/(tokens-1),
                       'completion_ms':1000*(stamps[name][-1]-start)}
                  for name,r in scheduler.requests.items()}
    finally:
        scheduler.close()
    snapshot=pool.snapshot()
    if snapshot['allocated_blocks'] or snapshot['request_handles']:
        raise RuntimeError('request cache leak')
    return {'duration_s':end-start,'total_output_tokens':2*tokens,
            'aggregate_tokens_per_s':2*tokens/(end-start),'requests':requests,
            'steps':steps,'pool_after':snapshot}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--device',choices=('cpu','npu'),default='npu')
    parser.add_argument('--dtype',choices=('float16','float32'),default='float16')
    parser.add_argument('--tokens',type=int,default=8)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--cache-dir',default='results/operator-cache')
    parser.add_argument('--output',default='results/m3-performance.json')
    args=parser.parse_args()
    if args.tokens<2 or args.tokens>256 or args.warmup<1 or args.runs<1:
        parser.error('tokens must be 2..256; warmup/runs must be positive')
    from pathlib import Path
    cache_dir=Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True,exist_ok=True)
    os.environ['ACL_OP_COMPILER_CACHE_MODE']='enable'
    os.environ['ACL_OP_COMPILER_CACHE_DIR']=str(cache_dir)
    report={'args':vars(args),'environment':environment_info(),'status':'running','passed':False,
            'cache_dir':str(cache_dir),'cache_mode':'enable','backends':{},
            'timing_scope':'Two identical requests arrive at common start; includes submit, queueing, scheduling, generation, release; excludes load/tokenize/pool creation/reporting. Host timestamps at step return; all tokens in a step share timestamp; not network streaming timing.'}
    def save(stage):
        report['stage']=stage
        write_report(args.output,report)
        print(stage,flush=True)
    try:
        torch,device,dtype=open_device(args.device,args.dtype)
        if device.type=='npu':
            torch.npu.set_option({'ACL_OP_COMPILER_CACHE_MODE':'enable','ACL_OP_COMPILER_CACHE_DIR':str(cache_dir)})
        from mini_llm_ascend.engine.weights import load_model
        from mini_llm_ascend.cache.paged_kv import PagedKVPool
        save('loading')
        report['provenance']=provenance(args.model)
        _,ids,_,eos=prepare_input(args.model,'请用一句话解释什么是KV缓存。',device)
        if ids.shape[1]+args.tokens>512: raise ValueError('capacity exceeded')
        report['input_ids']=ids[0].cpu().tolist()
        model=load_model(args.model,device=device,dtype=dtype)
        pool=PagedKVPool.for_model(model,32,16)
        expected=None
        def sample(mode,section):
            nonlocal expected
            result=run_trial(model,pool,ids,eos,torch,device,mode,args.tokens)
            outputs={name:r['output_ids'] for name,r in result['requests'].items()}
            if expected is None: expected=outputs
            if outputs!=expected: raise RuntimeError('serial/batched output mismatch')
            report['backends'][mode][section].append(result)
            print(mode,section,result['aggregate_tokens_per_s'],flush=True)
        for mode in ('serial','batched'):
            report['backends'][mode]={'warmups':[],'runs':[]}
            for i in range(args.warmup):
                save(mode+'_warmup_'+str(i+1)); sample(mode,'warmups'); save(mode+'_warmup_done')
        report['pairs']=[]
        for i in range(args.runs):
            order=['serial','batched'] if i%2==0 else ['batched','serial']
            report['pairs'].append(order)
            for mode in order:
                save(mode+'_run_'+str(i+1)); sample(mode,'runs'); save(mode+'_run_done')
        for backend in report['backends'].values():
            runs=backend['runs']
            backend['summary']={'sample_count':len(runs),
                'aggregate_tokens_per_s':statistics.mean(r['aggregate_tokens_per_s'] for r in runs),
                'pooled_tokens_per_s':sum(r['total_output_tokens'] for r in runs)/sum(r['duration_s'] for r in runs),
                'per_request':{name:{key:statistics.mean(r['requests'][name][key] for r in runs)
                   for key in ('ttft_ms','tpot_ms','completion_ms')} for name in ('a','b')}}
        report.update(status='ok',passed=True)
        code=0
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status='error',error=repr(exc)); code=1
    save('complete' if report['passed'] else 'failed')
    return code


if __name__=='__main__':
    raise SystemExit(main())
