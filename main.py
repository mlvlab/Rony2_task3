import os
import argparse
import json
import time
#task3
import task3.lib 
import task3.lib.predict as pred

#task2
from task2.lib.Task2 import *


def make_final_json(task2_answer):
    final_json = dict()
    final_json["task1_answer"] = ""
    final_json["task2_answer"] = task2_answer
    final_json["task3_answer"] = ""
    final_json["task4_answer"] = ""

    with open(args.json_path, 'w', encoding='utf-8') as make_file:
        json.dump(final_json, make_file, indent='\t')

def save_task3_json(task_num, set_num, pred_data, file_path=None):
    data, data_chk = {}, {}
    indent = 3
    
    if os.path.isfile(file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        indent += 1
    else:
        data[f"{task_num}"] = []
    
    data[f"{task_num}"].append({f"{set_num}": pred_data})

    with open(file_path, 'w') as json_file: 
        json.dump(data, json_file, indent=indent)
    with open(file_path, 'r') as json_file: 
        data_chk = json.load(json_file)
    print("{0} updated... > {1}".format(file_path, "OK" if data == data_chk else "Failed"))


def main(args):
    start = time.time()
    if os.path.exists(args.json_path) == True:
        os.remove(args.json_path)
    n_set = 5 if args.set_num == "all_sets" else 1
    
    # task_2 - 5 set 
    # data_path 나중에 대회 데이터셋 경로로 바꾸기 
    if args.run_single_task is None or args.run_single_task == 2:
        data_path = 'task2/temp_data'
        task2_answer = task2_inference(data_path)

    for i in range(n_set): #n_set : number of set
        args.set_num = f"set_0" + str(i+1) if n_set == 5 else args.set_num
        print("count : ", args.set_num)

        # task_1
        # todo
        if args.run_single_task is None or args.run_single_task == 1:
            pass

        # task_3
        if args.run_single_task is None or args.run_single_task == 3:
            t3_data = []
            t3 = pred.func_task3(args)
            t3_res_pred_move, t3_res_pred_stay, t3_res_pred_total = t3.run()
            t3_data.append(t3_res_pred_move)
            t3_data.append(t3_res_pred_stay)
            t3_data.append(t3_res_pred_total)

            # save result to json file
            save_task3_json(task_num="task3_answer", set_num=args.set_num, pred_data=t3_data, file_path=args.json_path)

        # task_4
        # todo
        if args.run_single_task is None or args.run_single_task == 4:
            pass
    
    if args.run_single_task is None or args.run_single_task == 2:
        make_final_json(task2_answer)

    print("TOTAL INFERENCE TIME : ", time.time()-start)

    
if __name__ == '__main__':
    p=argparse.ArgumentParser()
    # path
    # p.add_argument("--dataset_dir", type=str, default="/home/agc2021/dataset") # /set_01, /set_02, /set_03, /set_04, /set_05
    # p.add_argument("--root_dir", type=str, default="/home/[Team_ID]")
    # p.add_argument("--temporary_dir", type=str, default="/home/agc2021/temp")
    ###
    p.add_argument("--dataset_dir", type=str, default="../task3/dataset") # /set_01, /set_02, /set_03, /set_04, /set_05
    p.add_argument("--root_dir", type=str, default="./")
    p.add_argument("--temporary_dir", type=str, default="../task3/temp")
    
    ###
    p.add_argument("--json_path", type=str, default="answersheet_3_00_Rony2.json")
    p.add_argument("--task_num", type=str, default="task3_answer")
    p.add_argument("--set_num", type=str, default="all_sets") 
    p.add_argument("--device", default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    p.add_argument("--test", type = int, default = '3', help = 'number of video')

    p.add_argument('--run_single_task', type=int, default = None)
    p.add_argument("--release_mode", action='store_true', default=False)
    
    args = p.parse_args()

    assert(args.run_single_task in (1, 2, 3, 4, None))

    main(args)

    

