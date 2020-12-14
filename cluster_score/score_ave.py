import os
import csv
import numpy as np

# set the output dir path
code_path = os.path.dirname(os.path.realpath(__file__))

current_files = os.listdir(code_path)
current_files.sort(reverse = True)
# print(current_files)
# read_fold_name = '2020-Aug-26'
read_fold_name = current_files[3]   # 0:score_ave.py, 1: center.py, 2: Average folder
input_dir = os.path.join(code_path,read_fold_name)

ignore_name = ['log1']

output_dir = os.path.join(code_path,'Average')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

Error_csv = False
under_ave = False

def csv_write(output_list):
    global under_ave
    under_ave_count = 0
    csv_path = os.path.join(output_dir,read_fold_name+'.csv')
    # the first row of the csv file
    csv_first_row = ['File Name','Total Frames','Total Objects','Good Objects','Multiple Objects','No Objects','V measure score','Homogeneity','H(C|K)','H(C)','Completeness','H(K|C)','H(K)']
    ave_row =[]
    with open(csv_path,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_first_row)
        ave_row.append('Ave'+' ('+str(len(output_list))+' bags)')
        output_array = np.asarray(output_list)
        output_array = np.asarray(output_array[:,1:],dtype=float)
        col_sum = output_array.sum(axis=0)
        # print(col_sum)
        ave_row.append(str(int(col_sum[0])))
        ave_row.append(int(col_sum[1]))
        ave_row.append(int(col_sum[2]))
        ave_row.append(int(col_sum[3]))
        ave_row.append(int(col_sum[4]))
        col_sum = np.dot(output_array[:,0].reshape(1,len(output_array[:,0])),output_array[:,5:]) / int(col_sum[0])
        # print(col_sum)
        for i in range(len(col_sum[0])):
            ave_row.append(col_sum[0,i])
        writer.writerow(ave_row)
        for data in output_list:
            row = []
            for element in data:
                row.append(element)
            writer.writerow(row)
            # if float(data[6]) < float(col_sum[0,0]):
            #     if not under_ave:
            #         under_ave = True
            #         print('\033[1;94m'+'Below the average V-measure score'+'\033[0m')
            #     print(data[0])
                # under_ave_count += 1
    # print('\033[1;94m'+str(under_ave_count)+' files below the average (total '+str(len(output_list))+' files)'+'\033[0m')
    print(csv_first_row[0:7])
    print(ave_row[0:7])
    return

def ignore_file(filename):
    global ignore_name
    for ignore in ignore_name:
        if filename.find(ignore) != -1:
            return True
    return False

def csv_read():
    print("Reading directory:",input_dir)
    files = os.listdir(input_dir)
    files.sort(key=lambda x:x[:5],reverse = True)
    files.sort(key=lambda x:x[-8:])
    output_list = []
    global Error_csv
    for filename in files:
        if ignore_file(filename):
            continue
        csv_path = os.path.join(input_dir,filename)
        v_measure_score_list = []
        with open(csv_path,newline='') as csv_file:
            rows = csv.reader(csv_file)
            obj_num = []
            good_num = []
            multi_num = []
            no_num = []
            v_measure_score = []
            homo = []
            h_ck = []
            h_c = []
            comp = []
            h_kc = []
            h_k = []
            first = True
            for row in rows:
                if first:
                    first = False
                    continue
                if str(row).find('nan') != -1 or str(row).find('inf') != -1:
                    continue
                obj_num.append(int(row[1]))
                good_num.append(int(row[2]))
                multi_num.append(int(row[3]))
                no_num.append(int(row[4]))
                v_measure_score.append(float(row[5]))
                homo.append(float(row[6]))
                h_ck.append(float(row[7]))
                h_c.append(float(row[8]))
                comp.append(float(row[9]))
                h_kc.append(float(row[10]))
                h_k.append(float(row[11]))
            if len(v_measure_score) < 1:
                if not Error_csv:
                    Error_csv = True
                    print('\033[1;95m'+'Files Contain nan '+'\033[0m')
                print('\033[m'+str(filename)+'\033[0m')
                continue
            v_measure_score_list.append(str(filename))
            v_measure_score_list.append(len(v_measure_score))
            v_measure_score_list.append(sum(obj_num))
            v_measure_score_list.append(sum(good_num))
            v_measure_score_list.append(sum(multi_num))
            v_measure_score_list.append(sum(no_num))
            v_measure_score_list.append(sum(v_measure_score) / len(v_measure_score))
            v_measure_score_list.append(sum(homo) / len(homo))
            v_measure_score_list.append(sum(h_ck) / len(h_ck))
            v_measure_score_list.append(sum(h_c) / len(h_c))
            v_measure_score_list.append(sum(comp) / len(comp))
            v_measure_score_list.append(sum(h_kc) / len(h_kc))
            v_measure_score_list.append(sum(h_k) / len(h_k))
            output_list.append(v_measure_score_list)
    csv_write(output_list)
    return

if __name__ == '__main__':
    csv_read()