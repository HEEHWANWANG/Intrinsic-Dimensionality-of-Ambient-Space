import os
import pandas as pd
import copy

def save_result(result: dict,
                result_history: dict = None,
                result_file: str = './result.csv',
                verbose: bool = False):

    save_dict = result
    if verbose: print('save_result:', save_dict)

    if os.path.exists(result_file):
        result = pd.read_csv(result_file)
    else:
        result = pd.DataFrame(columns=save_dict.keys())

    if not result_history:
        result = result.append(save_dict, ignore_index=True)
    else:
        history_param = result_history.history.keys()
        for i in range(result_history.params['epochs']):
            save_dict_copy = copy.deepcopy(save_dict)
            save_dict_copy.update({'epoch': i})
            save_dict_copy.update({k:result_history.history[k][i] for k in history_param})
            result = result.append(save_dict_copy, ignore_index=True)


    result.to_csv(result_file, index=False)

    return
