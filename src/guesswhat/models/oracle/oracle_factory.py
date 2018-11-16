from guesswhat.models.oracle.oracle_film import FiLM_Oracle
from guesswhat.models.oracle.oracle_baseline import OracleNetwork


# stupid factory class to create networks

def create_oracle(config, num_words, reuse=False, device=''):

    network_type = config["type"]
    num_answers = 3

    if network_type == "film":
        return FiLM_Oracle(config, num_words, num_answers, device=device, reuse=reuse)
    elif network_type == "baseline":
        return OracleNetwork(config, num_words=num_words, num_answers=num_answers, device=device, reuse=reuse)
    else:
        assert False, "Invalid network_type: should be: baseline/cbn"


