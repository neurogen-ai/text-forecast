# src/mlflow/get_params.py 

def _get_params(
        params: dict[str, str | int],
        ignore_keys: list[str] = ['name', 'optimizer', 'last_epoch', 'step_count']
               ) -> dict[str, str | int]:
    """ Get params from dict, ignore clutter """
    return {k:v for k,v in params.items() if k not in ignore_keys and not str(k).startswith('_')}

def _get_scheduler_params(scheduler):
    """ Fetch and format scheduler (and sub-scheduler) parameters """
    params = {}
    params['name'] = type(scheduler).__name__
    if hasattr(scheduler, '_schedulers'):
        params['names'] = []
        params['milestones'] = scheduler._milestones
        for i, sub_scheduler in enumerate(scheduler._schedulers):
            params['names'].append(type(sub_scheduler).__name__)
            sub_params = _get_scheduler_params(sub_scheduler)
            params.update({f'{i}-{sub_params["name"]}-{k}':v for k,v in _get_params(sub_params).items()})
    else:
        params.update(_get_params(scheduler.__dict__))

    return params
