from pathlib import Path
import pickle
import sys
import time
import os
import traceback
import uuid
import datetime
import random
import shutil
import datetime
from typing import Dict, Iterable, Union, List

# Edge coverage. See https://github.com/ganler/tvm/tree/coverage
import git
import pandas as pd
import rich
from rich.progress import Progress, BarColumn, ProgressColumn
from rich.panel import Panel
from rich.console import RenderableType
from rich.columns import Columns

from nnsmith.abstract.op import ALL_OP_TYPES, auto_infer_in_dtypes, config_skip_op
from nnsmith import util
from nnsmith.error import IncorrectResult, NNSmithInternalError, SanityCheck
from nnsmith.graph_gen import GenerationTable
from nnsmith.backends import DiffTestBackend
from nnsmith.input_gen import gen_one_input
from nnsmith.difftest import assert_allclose
from nnsmith.graph_input_gen import forked_execution
import networkx as nx
import onnx
from summary import GraphSummary, ParamShapeSummary, SummaryBase

__COV_DRIVER__ = None

_METADATA_NAME_ = 'meta.txt'
_COV_BY_TIME_NAME_ = 'cov_by_time.csv'

# NOTE: Currently only engineered for TVM.


class Reporter:  # From Tzer.
    def __init__(self, report_folder=None, name_hint='') -> None:
        # Checks
        self.start_time = time.perf_counter()
        self.report_folder = report_folder

        if report_folder is None:
            self.report_folder = f'fuzzing-report-{uuid.uuid4()}'

        if os.path.exists(self.report_folder):
            # TODO: Allow continous fuzzing...
            decision = ''
            while decision.lower() not in ['y', 'n']:
                decision = input(
                    'Report folder already exists. Press [Y/N] to continue or exit...')
            if decision.lower() == 'n':
                raise NNSmithInternalError(
                    f'{self.report_folder} already exist... We want an empty folder to report...')
            else:
                shutil.rmtree(self.report_folder)

        os.mkdir(self.report_folder)
        print(f'Create report folder: {self.report_folder}')

        print(f'Using `{self.report_folder}` as the fuzzing report folder')
        with open(os.path.join(self.report_folder, _METADATA_NAME_), 'w') as f:
            fuzz_repo = git.Repo(search_parent_directories=True)

            def _log_repo(f, tag, repo: git.Repo):
                f.write(f'{tag} GIT HASH: {repo.head.object.hexsha}\n')
                f.write(f'{tag} GIT STATUS: ')
                f.write(
                    '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
                f.write(repo.git.status())
                f.write(
                    '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n')

            f.write(f'START TIME: {datetime.datetime.now()}')
            _log_repo(f, 'Fuzzer', fuzz_repo)
            if 'tvm' in name_hint and os.getenv('TVM_HOME'):
                _log_repo(f, 'TVM', git.Repo(os.getenv('TVM_HOME')))

        self.n_bug = 0
        self.record_coverage_cnt = 0

    def report_bug(self, err_type: Exception, buggy_onnx_path: str, buggy_torch_path: str, message: str, stdout: str, stderr: str, graph_path: str, sat_inputs=None):
        dir = f'{type(err_type).__name__}__{self.n_bug}'
        os.mkdir(os.path.join(self.report_folder, dir))
        G = pickle.load(open(graph_path, 'rb'))
        nx.drawing.nx_pydot.to_pydot(G).write_png(os.path.join(
            self.report_folder, dir, 'graph.png'))
        shutil.move(buggy_onnx_path, os.path.join(
            self.report_folder, dir, 'model.onnx'))
        if buggy_torch_path is not None:
            shutil.move(buggy_torch_path, os.path.join(
                self.report_folder, dir, 'model.pt'))
        shutil.move(stdout, os.path.join(
            self.report_folder, dir, 'stdout.log'))
        shutil.move(stderr, os.path.join(
            self.report_folder, dir, 'stderr.log'))
        for i in Path('.').glob('mlist.*'):
            shutil.move(str(i), os.path.join(
                self.report_folder, dir, i.stem))

        with open(os.path.join(self.report_folder, dir, 'err.txt'), 'w') as f:
            f.write(message)

        if sat_inputs is not None:
            pickle.dump(sat_inputs, open(os.path.join(
                self.report_folder, dir, 'sat_inputs.pkl'), 'wb'))

        self.n_bug += 1

    def flush(self, fuzz):
        if fuzz.table is not None:
            os.system('mv {} {}'.format(
                os.path.join(self.report_folder, f'state.pkl'),
                os.path.join(self.report_folder, f'state.pkl.bak')))
            pickle.dump({'table': fuzz.table}, open(
                os.path.join(self.report_folder, f'state.pkl'), 'wb'), protocol=4)
        profile = fuzz.profile  # type: pd.DataFrame
        if os.path.exists(os.path.join(self.report_folder, f'profile.pkl')):
            os.system('mv {} {}'.format(
                os.path.join(self.report_folder, f'profile.pkl'),
                os.path.join(self.report_folder, f'profile.pkl.bak')))
        profile.to_pickle(os.path.join(self.report_folder,
                          f'profile.pkl'), protocol=4)
        for i in fuzz.summaries:
            i.dump(os.path.join(self.report_folder, f'{i}.pkl'))

    def record_coverage(self, fuzz):
        if self.record_coverage_cnt % 10 == 0:
            self.flush(fuzz)
        self.record_coverage_cnt += 1
        with open(os.path.join(self.report_folder, _COV_BY_TIME_NAME_), 'a') as f:
            f.write(
                f'{time.perf_counter() - self.start_time :.2f},{__COV_DRIVER__.get_now()}\n')


class CustomProgress(Progress):
    def __init__(self, fuzz_status, columns: List[Union[str, ProgressColumn]]):
        self.fuzz_status = fuzz_status
        super().__init__(*columns)

    def get_renderables(self) -> Iterable[RenderableType]:
        """Get a number of renderables for the progress display."""
        yield self.fuzz_status()
        table = self.make_tasks_table(self.tasks)
        yield table


class FuzzingLoop:  # TODO: Support multiple backends.
    def __init__(self, backends: Dict[str, DiffTestBackend], mode='table', root=None, time_budget=60 * 60 * 4, max_nodes=32, inp_gen='random',
                 summaries: List[SummaryBase] = None, fork_bkn=False, _PER_MODEL_TIMEOUT_=1000):
        self.root = root
        self.reporter = Reporter(
            report_folder=root, name_hint=list(backends.keys())[0])
        self.mode = mode  # `random` or `table`
        self.inp_gen = inp_gen  # `random` or `grad`
        self.table = GenerationTable() if mode == 'table' else None

        SanityCheck.gt(len(backends), 0, "Empty backends are not allowed!")
        self.backends = backends

        self.time_budget = time_budget
        self.max_nodes = max_nodes

        self.cur_model_gen_t = float('nan')
        self.slowest_model_gen_t = -float("inf")
        self.fastest_model_gen_t = float("inf")

        self.cur_model_eval_t = float('nan')
        self.slowest_model_eval_t = -float("inf")
        self.fastest_model_eval_t = float("inf")

        self.cur_seed = 'nan'
        self.cur_node_size = 'nan'
        self.fork_bkn = fork_bkn

        self.stage = 'init'

        self._PER_MODEL_TIMEOUT_ = _PER_MODEL_TIMEOUT_  # milliseconds

        self.profile = pd.DataFrame(
            columns=['model_gen_t', 'model_eval_t', 'bugs', 'edge_cov'])
        self.summaries = summaries or []

        rich.print(
            f'[bold yellow]To exit the program: `kill {os.getpid()}`[/bold yellow]')
        rich.print(
            '[grey]This is because we use z3 written in C++ w/ Python wrappers. Ctrl+C may not stop it.')

    def rich(self):
        return Columns([
            Panel.fit(
                f'{datetime.timedelta(seconds=round(time.time()-self.start_time))} ~ '
                f'{datetime.timedelta(seconds=self.time_budget)}'
                f'\ncur_seed: {self.cur_seed}'
                f'\ncur_node_size: {self.cur_node_size}',
                title="Time Left ~ Total Time"),
            Panel.fit(f'{self.reporter.n_bug}/{len(self.profile)}'
                      f'\n{self.stage}',
                      title="Bug/Iter", style="magenta"),
            Panel.fit(f'[green]Fast: {self.fastest_model_gen_t:.3f}s[/green]|'
                      f'[bold]Cur: {self.cur_model_gen_t:.3f}s[/bold]\n'
                      f'[red]Slow: {self.slowest_model_gen_t:.3f}s[/red]|'
                      f'[red]Avg: {self.profile["model_gen_t"].mean():.3f}s',
                      title="Model Generation Time"),
            Panel.fit(f'[green]Fast: {self.fastest_model_eval_t:.3f}s[/green]|'
                      f'[bold]Cur: {self.cur_model_eval_t:.3f}s[/bold]\n'
                      f'[red]Slow: {self.slowest_model_eval_t:.3f}s[/red]|'
                      f'[red]Avg: {self.profile["model_eval_t"].mean():.3f}s',
                      title="Model Evaluation Time"),
        ])

    def fuzz(self):
        _TMP_ONNX_FILE_ = f'tmp_{uuid.uuid4()}.onnx'
        self.start_time = time.time()
        use_torch = any(i.__class__.__name__ ==
                        'TchExecutor' for i in self.backends.values())

        last_cov = 0

        try:
            with CustomProgress(
                fuzz_status=self.rich,
                columns=[
                    "[progress.description]{task.description}",
                    BarColumn(),
                    '[progress.percentage]{task.completed:>3.0f}/{task.total}',
                    '[progress.percentage]{task.percentage:>3.0f}%'],
            ) as progress:
                task_fuzz = progress.add_task(
                    '[green]Fuzzing time.', total=self.time_budget)
                task_coverage = progress.add_task(
                    '[green]Edge coverage.', total=__COV_DRIVER__.get_total())

                while True:
                    st = time.time()
                    self.reporter.record_coverage(self)
                    record_coverage_t = time.time() - st

                    gen_t_s = time.time()
                    while True:
                        try:
                            seed = random.getrandbits(32)
                            self.cur_seed = seed
                            self.cur_node_size = random.randint(
                                1, self.max_nodes)
                            self.stage = 'gen model'
                            progress.refresh()
                            sat_inputs, state, edge_set, seed = forked_execution(self.mode,
                                                                                 _TMP_ONNX_FILE_,
                                                                                 table=self.table,
                                                                                 max_node_size=self.cur_node_size,
                                                                                 max_gen_millisec=self._PER_MODEL_TIMEOUT_,
                                                                                 inp_gen=self.inp_gen,
                                                                                 save_torch=use_torch,
                                                                                 summaries=self.summaries,
                                                                                 seed=seed)
                            self.stage = 'load model'
                            load_t_s = time.time()
                            onnx_model = DiffTestBackend.get_onnx_proto(
                                _TMP_ONNX_FILE_)
                            load_model_time = time.time() - load_t_s
                            check_t_s = time.time()
                            self.stage = 'check model'
                            onnx.checker.check_model(
                                onnx_model, full_check=True)
                            check_model_time = time.time() - check_t_s
                            break
                        except Exception as e:
                            traceback.print_exc()
                            print('retrying...')

                    # Generation time logging.
                    self.cur_model_gen_t = time.time() - gen_t_s
                    self.fastest_model_gen_t = min(
                        self.fastest_model_gen_t, self.cur_model_gen_t)
                    self.slowest_model_gen_t = max(
                        self.slowest_model_gen_t, self.cur_model_gen_t)

                    eval_inputs = None
                    if sat_inputs is not None:
                        # sat_inputs is already a dict.
                        eval_inputs = sat_inputs
                    else:  # make some random inputs
                        input_spec, onames = DiffTestBackend.analyze_onnx_io(
                            onnx_model)
                        eval_inputs = gen_one_input(input_spec, 0, 1)

                    try:  # TODO: multi-process support for isolation.
                        eval_t_s = time.time()
                        info = {}

                        if use_torch:
                            torch_model = pickle.load(
                                open(_TMP_ONNX_FILE_ + '.pt', 'rb'))
                        else:
                            torch_model = None

                        difftest_pool = {}
                        with util.stdout_redirected(f"{_TMP_ONNX_FILE_}.stdout", sys.__stdout__), \
                                util.stdout_redirected(f"{_TMP_ONNX_FILE_}.stderr", sys.__stderr__):
                            for bname in self.backends:
                                self.stage = f'eval {bname}'
                                st = time.time()
                                difftest_pool[bname] = self.backends[bname].predict(
                                    onnx_model, eval_inputs, torch_model=torch_model)
                                info['model_eval_t_' + bname] = time.time() - st
                        self.stage = f'diff testing'

                        keys = list(difftest_pool.keys())
                        for idx in range(1, len(keys)):
                            assert_allclose(
                                difftest_pool[keys[0]],
                                difftest_pool[keys[idx]],
                                keys[0], keys[idx],
                                nan_as_err=False,
                                safe_mode=True)

                        self.stage = f'cleanup'
                        # Evaluation time logging.
                        self.cur_model_eval_t = time.time() - eval_t_s
                        self.fastest_model_eval_t = min(
                            self.fastest_model_eval_t, self.cur_model_eval_t)
                        self.slowest_model_eval_t = max(self.slowest_model_eval_t,
                                                        self.cur_model_eval_t)

                        cur_cov = __COV_DRIVER__.get_now()
                        if edge_set:
                            for src, dst in edge_set:
                                if cur_cov == last_cov:
                                    self.table.on_no_cov(src, dst)
                                else:
                                    self.table.on_new_cov(src, dst)

                    except Exception as e:
                        self.stage = f'reporting bug'
                        # ignore models with invalid inputs
                        if not isinstance(e, IncorrectResult) or sat_inputs is not None:
                            stdout = f'{_TMP_ONNX_FILE_}.stdout'
                            stderr = f'{_TMP_ONNX_FILE_}.stderr'
                            graph = f'{_TMP_ONNX_FILE_}-graph.pkl'
                            torch_path = f'{_TMP_ONNX_FILE_}.pt' if use_torch else None
                            self.reporter.report_bug(
                                e, _TMP_ONNX_FILE_, torch_path, str(e), stdout, stderr, graph, sat_inputs=sat_inputs)

                    summaries_st = time.time()
                    graph = pickle.load(
                        open(_TMP_ONNX_FILE_ + '-graph.pkl', 'rb'))
                    for i in self.summaries:
                        i.update(graph)
                    summaries_t = time.time() - summaries_st
                    cur_time = time.time()
                    progress.update(
                        task_fuzz, completed=cur_time - self.start_time)
                    progress.update(
                        task_coverage, completed=__COV_DRIVER__.get_now())
                    info.update({
                        'model_gen_t': self.cur_model_gen_t,
                        'model_eval_t': self.cur_model_eval_t,
                        'edge_cov': __COV_DRIVER__.get_now(),
                        'bugs': self.reporter.n_bug,
                        'time_stamp': time.perf_counter() - self.start_time,
                        'summaries_update_time': summaries_t,
                        'record_coverage_time': record_coverage_t,
                        'seed': seed,
                        'node_size': self.cur_node_size,
                        'load_model_time': load_model_time,
                        'check_model_time': check_model_time,
                    })
                    for s in self.summaries:
                        info.update(
                            {'s_' + k: v for k, v in s.report().items()})
                    self.profile = self.profile.append(info, ignore_index=True)

                    if cur_time - self.start_time > self.time_budget:
                        break
        finally:  # cleanup
            os.system('rm ' + _TMP_ONNX_FILE_ + '*')
            last_cov = __COV_DRIVER__.get_now()
        self.reporter.flush(self)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./fuzz_report')
    parser.add_argument('--time_budget', type=int, default=60 * 60 * 4)
    parser.add_argument('--backend', type=str, default='tvm')
    parser.add_argument('--mode', type=str, default='table')
    parser.add_argument(
        '--skip', help='Node types to skip. Split by `,`. By default a blacklist for each backend is also appended.', type=str)
    parser.add_argument('--inp_gen', type=str, default='random')
    parser.add_argument('--gen_timeout', type=int,
                        default=1000, help='in milliseconds')
    args = parser.parse_args()

    backends = None
    if args.backend == 'tvm':
        from nnsmith.backends.tvm_graph import TVMExecutor
        backends = {'tvm-opt': TVMExecutor(opt_level=4),
                    'tvm-debug': TVMExecutor(opt_level=0)}
        __COV_DRIVER__ = TVMExecutor.coverage_install()
    elif args.backend == 'ort':
        from nnsmith.backends.ort_graph import ORTExecutor
        backends = {'ort-opt': ORTExecutor(opt_level=3),
                    'ort-debug': ORTExecutor(opt_level=0)}
        __COV_DRIVER__ = ORTExecutor.coverage_install()
    elif args.backend == 'trt':
        from nnsmith.backends.trt_graph import TRTBackend
        from nnsmith.backends.tch_graph import TchExecutor
        backends = {'trt-opt': TRTBackend(),
                    'tch-debug': TchExecutor(opt_level=0, dev='cpu')}
        __COV_DRIVER__ = TRTBackend.coverage_install()
    elif args.backend == 'tch':
        from nnsmith.backends.tch_graph import TchExecutor
        backends = {'tch-opt': TchExecutor(dev='cuda'),
                    'tch-debug': TchExecutor(opt_level=0, dev='cpu')}
        __COV_DRIVER__ = TchExecutor.coverage_install()
    else:
        raise NotImplementedError("Other backends not supported yet.")
    skip = 'backend:' + args.backend
    if args.skip is not None:
        skip += ',' + args.skip
    auto_infer_in_dtypes()
    config_skip_op(skip)
    fuzzing_loop = FuzzingLoop(
        root=args.root,
        backends=backends,
        mode=args.mode,
        time_budget=args.time_budget,
        inp_gen=args.inp_gen,
        summaries=[ParamShapeSummary(), GraphSummary(), GraphSummary(level=1)],
        _PER_MODEL_TIMEOUT_=args.gen_timeout
    )
    fuzzing_loop.fuzz()
