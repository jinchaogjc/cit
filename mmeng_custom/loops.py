import copy
import logging
import mmengine
import torch
import torch.nn as nn

from typing import Callable, List, Union
from mmengine import LOOPS, print_log
from mmengine.runner import IterBasedTrainLoop, ValLoop
from mmengine.model import MMDistributedDataParallel

from .Scenario import Scenario

def set_model_attr(model, attr_name, attr_value):
    setattr(model, attr_name, attr_value)
    for child in model.children():
        set_model_attr(child, attr_name, attr_value)
    pass

@LOOPS.register_module()
class ScenarioBasedTrainLoop(IterBasedTrainLoop):
    def __init__(self, **kwargs):
        self._dataloader_cfg = copy.deepcopy(kwargs['dataloader'])
        self.scenario = Scenario(
            **kwargs.pop('scenario', dict())
        )
        super().__init__(**kwargs)

        diff_rank_seed = self.runner._randomness_cfg.get(
            'diff_rank_seed', False)
        self.train_loaders, self.train_loader_iterators = self.scenario.generate_dataloader_by_scenario(
            self.dataloader.dataset,
            seed=self._runner.seed,
            diff_rank_seed=diff_rank_seed,
            dataloader_cfg=self._dataloader_cfg,
            build_dataloader=self._runner.build_dataloader,
            test=False
        )

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        root_dir = self.runner.work_dir
        for task_id in range(self.scenario.start_task, self.scenario.nb_tasks):
            self.before_scenario(root_dir, task_id)
            # reset the checkpoint hook
            self.runner._hooks[-1].out_dir = None
            self.runner._hooks[-1].before_train(self.runner)
            # self.reset_learning_state(self.runner, task_id)

            print_log(f"=============Task {task_id} starts==============",
                      logger='current',
                      level=logging.WARNING)
            
            self.run_Scenario(task_id)
            print_log(f"=============Task {task_id} ends==============",
                      logger='current',
                      level=logging.WARNING)

            self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')

        return self.runner.model
    
    def run_Scenario(self, task_id):
        current_pre_cls = self.scenario.initial_increment + task_id * self.scenario.increment
        current_total_cls = current_pre_cls + self.scenario.increment
        if isinstance(self.runner.model, MMDistributedDataParallel):
            self.runner.model.module.decode_head.num_classes = current_total_cls
            self.runner.model.module.pre_num_cls = current_pre_cls
            self.runner.model.module.new_num_cls = self.scenario.increment
            self.runner.model.module.decode_head.loss_decode.criterion.ignore_range = current_pre_cls
            if not self.runner.model.module.freeze_backbone:
                setattr(self.runner.model.module,
                                'backbone_freeze_{}'.format(task_id),
                                copy.deepcopy(self.runner.model.module.backbone))
                for param in getattr(self.runner.model.module, 'backbone_freeze_{}'.format(task_id)).parameters():
                    param.requires_grad = False

            setattr(self.runner.model.module,
                                'decode_head_freeze_{}'.format(task_id),
                                copy.deepcopy(self.runner.model.module.decode_head))
            
            for param in getattr(self.runner.model.module, 'decode_head_freeze_{}'.format(task_id)).parameters():
                param.requires_grad = False
            # then update the model structure
            pre_q = self.runner.model.module.decode_head.q
            assert pre_q.weight.shape[0] == current_pre_cls
            new_q = nn.Embedding(current_total_cls, self.runner.model.module.decode_head.channels).to(pre_q.weight.device)
            new_q.weight.data[:current_pre_cls].copy_(pre_q.weight.data)
            # new_q.weight.data[current_pre_cls:] *= 0
            new_q.requires_grad_(True)
            self.runner.model.module.decode_head.q = new_q
            # set for evaluation
            self.runner.model.module.out_channels = current_total_cls
            # then update the optimizer
            self.reset_learning_state(self.runner, task_id)
        else:
            self.runner.model.decode_head.num_classes = current_total_cls
            self.runner.model.pre_num_cls = current_pre_cls
            self.runner.model.new_num_cls = self.scenario.increment
            self.runner.model.decode_head.loss_decode.criterion.ignore_range = current_pre_cls
            # first establish the freeze branch
            if not self.runner.model.freeze_backbone:
                setattr(self.runner.model,
                                'backbone_freeze_{}'.format(task_id),
                                copy.deepcopy(self.runner.model.backbone))
                for param in getattr(self.runner.model, 'backbone_freeze_{}'.format(task_id)).parameters():
                    param.requires_grad = False
                    
            setattr(self.runner.model,
                                'decode_head_freeze_{}'.format(task_id),
                                copy.deepcopy(self.runner.model.decode_head))
            
            for param in getattr(self.runner.model, 'decode_head_freeze_{}'.format(task_id)).parameters():
                param.requires_grad = False
            # then update the model structure
            pre_q = self.runner.model.decode_head.q
            assert pre_q.weight.shape[0] == current_pre_cls
            new_q = nn.Embedding(current_total_cls, self.runner.model.decode_head.channels).to(pre_q.weight.device)
            new_q.weight.data[:current_pre_cls].copy_(pre_q.weight.data)
            # new_q.weight.data[current_pre_cls:] *= 0
            new_q.requires_grad_(True)
            self.runner.model.decode_head.q = new_q
            # set for evaluation
            self.runner.model.out_channels = current_total_cls
            # then update the optimizer
            self.reset_learning_state(self.runner, task_id)

        torch.cuda.empty_cache()

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()
            self.dataloader_iterator = self.train_loader_iterators[task_id]
            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and self._iter % self.val_interval == 0):
                self.runner.val_loop.run()

        # print()
    
    # renitialize the model and dataloader
    def before_scenario(self, root_dir, task_id):
        self._iter = 0
        setattr(self.runner, 'current_task', task_id)
        setattr(self.runner, '_work_dir', root_dir + f'/task_{task_id}')
        mmengine.mkdir_or_exist(self.runner._work_dir)
        set_model_attr(self.runner.model, 'current_task', task_id)
        # set_model_attr(self.runner.model, 'increment', self.scenario.increment)
        # set_model_attr(self.runner.model, 'super_runner', self.runner)

    def reset_learning_state(self, runner, task_id):
        self.dataloader = self.train_loaders[task_id]
        runner.optim_wrapper = runner.build_optim_wrapper(runner.cfg['optim_wrapper'])
        runner.param_schedulers = runner.build_param_scheduler(runner.cfg['param_scheduler'])

        runner.model = runner.wrap_model(
            runner.cfg.get('model_wrapper_cfg'), runner.model)
        runner.optim_wrapper.initialize_count_status(
            runner.model,
            0,  # type: ignore
            self.max_iters)  # type: ignore

@LOOPS.register_module()
class ScenarioBasedValLoop(ValLoop):
    def __init__(self, **kwargs):
        self._dataloader_cfg = copy.deepcopy(kwargs['dataloader'])
        self.scenario = Scenario(
            **kwargs.pop('scenario', dict())
        )
        super().__init__(**kwargs)

        diff_rank_seed = self.runner._randomness_cfg.get(
            'diff_rank_seed', False)
        self.val_loaders = self.scenario.generate_dataloader_by_scenario(
            self.dataloader.dataset,
            seed=self._runner.seed,
            diff_rank_seed=diff_rank_seed,
            dataloader_cfg=self._dataloader_cfg,
            build_dataloader=self._runner.build_dataloader,
            test=True
        )[0]

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        self.dataloader = self.val_loaders[self.runner.current_task]
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics