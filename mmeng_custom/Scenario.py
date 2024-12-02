import copy
from typing import Callable, List, Union
from mmengine.runner.loops import _InfiniteDataloaderIterator

class Scenario:
    def __init__(
            self,
            start_task: int = 0,
            nb_tasks: int = 0,
            increment: Union[List[int], int] = 0,
            initial_increment: int = 0,
            # class_order: Union[List[int], None] = None,
            num_classes: int = 100,
            ) -> None:
        self.start_task = start_task # this used to start from the middle of the training
        self.nb_tasks = nb_tasks
        self.increment = increment
        self.initial_increment = initial_increment
        # self.class_order = class_order
        self.num_classes = num_classes

    def generate_dataloader_by_scenario(self,
                                        dataset,
                                        seed,
                                        diff_rank_seed,
                                        dataloader_cfg,
                                        build_dataloader,
                                        test=False,
                                        ):
        total_tasks = (self.num_classes - self.initial_increment) // self.increment

        dataloaders = []
        dataloader_iterators = []
        for task_id in range(total_tasks):
            _dataset = copy.deepcopy(dataloader_cfg['dataset'])
            # only the class_range is changed
            start = 1 if test else task_id*self.increment+self.initial_increment + 1
            end = (task_id+1)*self.increment+self.initial_increment
            for transforms in _dataset['pipeline']:
                if transforms['type'] == 'LoadAnnotationsCL':
                    transforms['class_range'] = [start, end]
                    break

            dataloader = copy.deepcopy(dataloader_cfg)
            dataloader['dataset'] = _dataset
            #     'diff_rank_seed', False)
            dataloader = build_dataloader(
                dataloader,
                seed=seed,
                diff_rank_seed=diff_rank_seed
            )
            dataloader_iterator = _InfiniteDataloaderIterator(dataloader)

            dataloaders.append(dataloader)
            dataloader_iterators.append(dataloader_iterator)

        return dataloaders, dataloader_iterators
