import json
import random
import logging
import os

from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

_LOGGER = logging.getLogger(__name__)
_STATE_PATH = "./toy-state.json"


@record
def main():
    logging.basicConfig(level=logging.INFO)

    dist.init_process_group()

    rank = dist.get_rank()
    local_rank = os.environ["LOCAL_RANK"]
    world_size = dist.get_world_size()

    _LOGGER.info(f"local_rank={local_rank} rank={rank} world size={world_size}")

    state = {"num_steps": 0}
    if os.path.exists(_STATE_PATH):
        with open(_STATE_PATH) as fp:
            state = json.load(fp)

    random.seed(rank + world_size * state["num_steps"])

    while True:
        value = random.random()
        _LOGGER.info(f"[{rank=}] step={state['num_steps']} {value=}")
        if value < 0.001:
            raise ValueError("Encountered fake bad value.")

        state["num_steps"] += 1

        dist.barrier()
        if rank == 0:
            with open(_STATE_PATH, "w") as fp:
                json.dump(state, fp)
        dist.barrier()


if __name__ == "__main__":
    main()
