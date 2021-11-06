import os
import numpy as np

def convert_intsec_to_strsec(sec):
    if sec == None:
        strsec = 'NONE'
    else:
        min_str = str(sec // 60)
        sec_str = str(sec % 60)
        strsec = min_str.zfill(2) + ':' + sec_str.zfill(2)
    return strsec
# print(convert_intsec_to_strsec(251))

def answer_list_to_json(answer_list):
    # answer_list_shape : set_num(# of testset) * drone_num(3) * class_num(3) * instance_num(?)
    
    set_num = len(answer_list)

    out_str = "{\n"
    out_str += '    "task2_answer": [{\n'
    for i in range(set_num):
        out_str += '        "set_' + str(i+1) + '":[\n'
        for j in range(3):
            out_str += '            {drone_' + str(j+1) + ':[{\n'
            out_str += '                "M": ['
            for k in range(len(answer_list[i][j][0])):
                ans = answer_list[i][j][0][k]
                out_str += '"' + convert_intsec_to_strsec(ans) + '"'
                if k != len(answer_list[i][j][0]) - 1:
                    out_str += ','
            out_str += '],\n'
            out_str += '                "W": ['
            for k in range(len(answer_list[i][j][1])):
                ans = answer_list[i][j][1][k]
                out_str += '"' + convert_intsec_to_strsec(ans) + '"'
                if k != len(answer_list[i][j][1]) - 1:
                    out_str += ','
            out_str += '],\n'
            out_str += '                "C": ['
            for k in range(len(answer_list[i][j][2])):
                ans = answer_list[i][j][2][k]
                out_str += '"' + convert_intsec_to_strsec(ans) + '"'
                if k != len(answer_list[i][j][2]) - 1:
                    out_str += ','
            out_str += ']}]}'
            if j != 2:
                out_str += ',\n'
            elif (i == set_num - 1) and (j == 2):
                out_str += ']\n'
            else:
                out_str += '],\n'
        if i == set_num -1 :
            out_str += '    }]\n'
            out_str += '}'
    return out_str

def cw_metrics_cal(gt, pred):
    correct = 0
    deletion = 0
    insertion = 0
    include_list = []
    for ii in range(len(gt)):
        is_include = 0
        if gt[ii][0] == None:
            if pred[0] == None:
                correct += 1
        else:
            if pred[0] == None:
                deletion += 1
            else:
                gt_start, gt_end = gt[ii][0], gt[ii][1]
                for jj in range(len(pred)):
                    if gt_start <= pred[jj] and pred[jj] <= gt_end:
                        include_list.append(pred[jj])
                        is_include += 1
                if is_include == 0:
                    deletion += 1
                elif is_include > 1:
                    insertion = is_include - 1
                elif is_include == 1:
                    correct += 1
    substitution = len(pred) - len(list(set(include_list)))
#     print('s, d, i, correct:', substitution, deletion, insertion, correct)
    return substitution, deletion, insertion, correct

def evaluation(gt_list, answer_list):
    set_num = len(gt_list)
    total_s, total_d, total_i, total_n, total_correct = 0, 0, 0, 0, 0
    for i in range(set_num):
        sw_s, sw_d, sw_i, sw_n, sw_correct = 0, 0, 0, 0, 0
        for j in range(3): # for drones 1 ~ 3
            dw_s, dw_d, dw_i, dw_n, dw_correct = 0, 0, 0, 0, 0
            for k in range(3): # for class man, woman, child
                cw_s, cw_d, cw_i, cw_correct = cw_metrics_cal(gt_list[i][j][k], answer_list[i][j][k])
                dw_s += cw_s
                dw_d += cw_d
                dw_i += cw_i
                dw_n += len(gt_list[i][j][k])
                dw_correct += cw_correct
            dw_er = (dw_s + dw_d + dw_i) / dw_n
            print('Set', str(i), 'Drone', str(j), 's, d, i, er, correct:', dw_s, dw_d, dw_i, np.round(dw_er, 2), dw_correct)
            sw_s += dw_s
            sw_d += dw_d
            sw_i += dw_i
            sw_n += dw_n
            sw_er = (sw_s + sw_d + sw_i) / sw_n
            sw_correct += dw_correct
        total_s += sw_s
        total_d += sw_d
        total_i += sw_i
        total_n += sw_n
        total_er = (total_s + total_d + total_i) / total_n
        total_correct += sw_correct
        print('Subtotal Set', str(i), 's, d, i, er, correct:', sw_s, sw_d, sw_i, np.round(sw_er, 2), sw_correct)
    print('Total', 's, d, i, er, correct:', total_s, total_d, total_i, np.round(total_er, 2), total_correct)
    return total_s, total_d, total_i, total_er, total_correct

if __name__ == '__main__':
    # answer_list_shape : set_num(# of testset) * drone_num(3) * class_num(3) * instance_num(?)

    # test json writing
    answer_list_for_test_json =   [
                        [
                            [[33, 38, 59, 80], [33, 48, 62], [10]],
                            [[5, 48], [33, 53], [60, 73]],
                            [[None],[10, 42, 230], [None]]
                        ],
                        [
                            [[200, 228], [72, 225, 252], [None]],
                            [[None], [None], [7, 112]],
                            [[260],[None], [161, 238]]
                        ],
                    ]
    out_str = answer_list_to_json(answer_list_for_test_json)                    
    print(out_str)


    # test evaluation
    gt_list = [   # gt_list for 세부문제정의서 예시 (1 set, 1 drone, 3 class)
                        [
                            [[[5, 12]], [[0, 7], [13, 20]], [[0, 3], [11, 14]]],
                            [[[5, 12]], [[0, 7], [13, 20]], [[0, 3], [11, 14]]],
                            [[[5, 12]], [[0, 7], [13, 20]], [[0, 3], [11, 14]]],                        
                        ],
            ]
    answer_list = [
                        [
                            [[2, 6], [13, 18], [19]],
                            [[2, 6], [13, 18], [19]],
                            [[2, 6], [13, 18], [19]],
                        ],
            ]

    gt_list_for_samples = [   # gt_list for sample video (1 set, 3 drone, 3 class)
                        [
                            [[[210, 214], [217, 220]], [[222, 226]], [[74, 79], [225, 231]]],
                            [[[168, 172], [183, 186]], [[175, 179]], [[164, 169]]],
                            [[[213, 216], [220, 224]], [[214, 218]], [[226, 231]]],
                        ],                                                                                              
            ]

    answer_list_for_samples = [
                        [
                            [[213, 218], [222], [75, 228]],
                            [[172, 184], [175], [169]],
                            [[215, 222], [218], [226]],
                        ],
            ]

    print('# Evaluation for 세부문제정의서 샘플')
    evaluation(gt_list, answer_list)

    print('\n\n# Evaluation for Sample Video')
    evaluation(gt_list_for_samples, answer_list_for_samples)


    # gt_list_for_samples = [   # gt_list for sample video (1 set, 3 drone, 3 class)
    #                     [
    #                         [[[210, 214], [217, 220]], [[222, 226]], [[74, 79], [225, 231]]],
    #                         [[[168, 172], [183, 186]], [[175, 179]], [[164, 169]]],
    #                         [[[213, 216], [220, 224]], [[214, 218]], [[226, 231]]],
    #                     ],
    #                     [
    #                         [[[210, 214], [217, 220]], [[222, 226]], [[74, 79], [225, 231]]],
    #                         [[[168, 172], [183, 186]], [[175, 179]], [[164, 169]]],
    #                         [[[213, 216], [220, 224]], [[214, 218]], [[226, 231]]],
    #                     ],
    #                     [
    #                         [[[210, 214], [217, 220]], [[222, 226]], [[74, 79], [225, 231]]],
    #                         [[[168, 172], [183, 186]], [[175, 179]], [[164, 169]]],
    #                         [[[213, 216], [220, 224]], [[214, 218]], [[226, 231]]],
    #                     ],
    #                     [
    #                         [[[210, 214], [217, 220]], [[222, 226]], [[74, 79], [225, 231]]],
    #                         [[[168, 172], [183, 186]], [[175, 179]], [[164, 169]]],
    #                         [[[213, 216], [220, 224]], [[214, 218]], [[226, 231]]],
    #                     ],
    #                     [
    #                         [[[210, 214], [217, 220]], [[222, 226]], [[74, 79], [225, 231]]],
    #                         [[[168, 172], [183, 186]], [[175, 179]], [[164, 169]]],
    #                         [[[213, 216], [220, 224]], [[214, 218]], [[226, 231]]],
    #                     ],                                                                                                
    #         ]