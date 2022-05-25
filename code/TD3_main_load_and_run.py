import json
from TD3.Training import Training_DDPG as training

from pathlib import Path


def main():

    save_path = 'TD3/saves/last_eval/prio_without_annealing/random_opp_equal_weighted/'
    save_path = save_path + 'BASIC_OPP/best_weights'
    store_eval(training.eval(weak=True, weights_path=save_path), save_path + '/weak_opp')
    store_eval(training.eval(weak=False, weights_path=save_path), save_path + '/strong_opp')


def store_eval(eval, path_with_run_no):
    Path(path_with_run_no).mkdir(parents=True, exist_ok=True)
    with open(path_with_run_no + "/eval.json", "w") as f:
        # Write it to file
        json.dump(eval, f)

if __name__ == '__main__':
    main()

# python TD3_main_load_and_run.py -w True -p 'TD3/saves/final_run_01/DEFENSE/weights' -s 250 -n 100 --show False -d 0.0

