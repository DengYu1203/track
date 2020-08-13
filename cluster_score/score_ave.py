import os
import csv
import numpy as np

# set the output dir path
code_path = os.path.dirname(os.path.realpath(__file__))

read_fold_name = '2020-Aug-13'
input_dir = os.path.join(code_path,read_fold_name)

output_dir = os.path.join(code_path,'Average')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def csv_write(output_list):
    csv_path = os.path.join(output_dir,read_fold_name+'.csv')
    with open(csv_path,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file name','total frames','V measure score','Homogeneity','H(C|K)','H(C)','Completeness','H(K|C)','H(K)']) # the first row of the csv file
        for data in output_list:
            row = []
            for element in data:
                row.append(element)
            writer.writerow(row)
        row =[]
        row.append('Ave')
        output_array = np.asarray(output_list)
        output_array = np.asarray(output_array[:,1:],dtype=float)
        col_sum = output_array.sum(axis=0)
        row.append(int(col_sum[0]))
        col_sum = np.dot(output_array[:,0].reshape(1,len(output_array[:,0])),output_array[:,1:]) / int(col_sum[0])
        # print(col_sum[0])
        for i in range(len(col_sum[0])):
            row.append(col_sum[0,i])
        writer.writerow(row)
    return

def csv_read():
    files = os.listdir(input_dir)
    files.sort()
    output_list = []
    for filename in files:
        csv_path = os.path.join(input_dir,filename)
        v_measure_score_list = []
        v_measure_score_list.append(str(filename))
        with open(csv_path,newline='') as csv_file:
            rows = csv.reader(csv_file)
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
                if str(row).find('nan') != -1:
                    continue
                v_measure_score.append(float(row[1]))
                homo.append(float(row[2]))
                h_ck.append(float(row[3]))
                h_c.append(float(row[4]))
                comp.append(float(row[5]))
                h_kc.append(float(row[6]))
                h_k.append(float(row[7]))
            v_measure_score_list.append(len(v_measure_score))
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