import os
import csv
import numpy as np
from datetime import datetime

# set the output dir path
code_path = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(code_path,'radar_scenes','f1score')
today = datetime.today().strftime('%Y-%b-%d')

output_dir = os.path.join(code_path,'radar_scenes','Average')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

Error_csv = False
under_ave = False
ignore_name = []
validation_set = ['sequence_107','sequence_135','sequence_14','sequence_24','sequence_53','sequence_68','sequence_85',
                  'sequence_111','sequence_138','sequence_153','sequence_31','sequence_58','sequence_6','sequence_89',
                  'sequence_122','sequence_147','sequence_155','sequence_42','sequence_5','sequence_73','sequence_93',
                  'sequence_130','sequence_148','sequence_19','sequence_48','sequence_63','sequence_79','sequence_99']

def csv_write(output_list,output_list_val):
    global under_ave
    under_ave_count = 0
    csv_path = os.path.join(output_dir,today+'_f1score.csv')
    # the first row of the csv file
    csv_first_row = ['File Name','F1-score(IOU>=0.3)','Precision','Recall','TP','FP','FN','F1-score(IOU>=0.5)','Precision','Recall','TP','FP','FN']
    ave_row =[]
    ave_row_val = []
    with open(csv_path,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_first_row)
        """
        Validation Data
        """
        ave_row_val.append('Val'+' ('+str(len(output_list_val))+' bags)')
        output_val_array = np.asarray(output_list_val)
        output_val_array = np.asarray(output_val_array[:,1:],dtype=float)
        col_val_sum = output_val_array.sum(axis=0)
        # f1 score(iou>=0.3), precision, recall
        ave_row_val.append(float(2*col_val_sum[0]/(2*col_val_sum[0]+col_val_sum[1]+col_val_sum[2])))
        ave_row_val.append(float(col_val_sum[0]/(col_val_sum[0]+col_val_sum[1])))
        ave_row_val.append(float(col_val_sum[0]/(col_val_sum[0]+col_val_sum[2])))
        ave_row_val.append(int(col_val_sum[0]))
        ave_row_val.append(int(col_val_sum[1]))
        ave_row_val.append(int(col_val_sum[2]))
        # f1 score(iou>=0.5), precision, recall
        ave_row_val.append(float(2*col_val_sum[3]/(2*col_val_sum[3]+col_val_sum[4]+col_val_sum[5])))
        ave_row_val.append(float(col_val_sum[3]/(col_val_sum[3]+col_val_sum[4])))
        ave_row_val.append(float(col_val_sum[3]/(col_val_sum[3]+col_val_sum[5])))
        ave_row_val.append(int(col_val_sum[3]))
        ave_row_val.append(int(col_val_sum[4]))
        ave_row_val.append(int(col_val_sum[5]))
        writer.writerow(ave_row_val)
        """
        Training Data
        """
        ave_row.append('Train'+' ('+str(len(output_list))+' bags)')
        output_array = np.asarray(output_list)
        output_array = np.asarray(output_array[:,1:],dtype=float)
        col_sum = output_array.sum(axis=0)
        # f1 score(iou>=0.3), precision, recall
        ave_row.append(float(2*col_sum[0]/(2*col_sum[0]+col_sum[1]+col_sum[2])))
        ave_row.append(float(col_sum[0]/(col_sum[0]+col_sum[1])))
        ave_row.append(float(col_sum[0]/(col_sum[0]+col_sum[2])))
        ave_row.append(int(col_sum[0]))
        ave_row.append(int(col_sum[1]))
        ave_row.append(int(col_sum[2]))
        # f1 score(iou>=0.5), precision, recall
        ave_row.append(float(2*col_sum[3]/(2*col_sum[3]+col_sum[4]+col_sum[5])))
        ave_row.append(float(col_sum[3]/(col_sum[3]+col_sum[4])))
        ave_row.append(float(col_sum[3]/(col_sum[3]+col_sum[5])))
        ave_row.append(int(col_sum[3]))
        ave_row.append(int(col_sum[4]))
        ave_row.append(int(col_sum[5]))
        writer.writerow(ave_row)
        # Validation Data
        for data in output_list_val:
            row = []
            row.append(data[0])
            row.append(float(2*data[1]/(2*data[1]+data[2]+data[3])))
            row.append(float(data[1]/(data[1]+data[2])))
            row.append(float(data[1]/(data[1]+data[3])))
            row.append(int(data[1]))
            row.append(int(data[2]))
            row.append(int(data[3]))
            row.append(float(2*data[4]/(2*data[4]+data[5]+data[6])))
            row.append(float(data[4]/(data[4]+data[5])))
            row.append(float(data[4]/(data[4]+data[6])))
            row.append(int(data[4]))
            row.append(int(data[5]))
            row.append(int(data[6]))
            writer.writerow(row)
        # Training Data
        for data in output_list:
            row = []
            row.append(data[0])
            row.append(float(2*data[1]/(2*data[1]+data[2]+data[3])))
            row.append(float(data[1]/(data[1]+data[2])))
            row.append(float(data[1]/(data[1]+data[3])))
            row.append(int(data[1]))
            row.append(int(data[2]))
            row.append(int(data[3]))
            row.append(float(2*data[4]/(2*data[4]+data[5]+data[6])))
            row.append(float(data[4]/(data[4]+data[5])))
            row.append(float(data[4]/(data[4]+data[6])))
            row.append(int(data[4]))
            row.append(int(data[5]))
            row.append(int(data[6]))
            writer.writerow(row)

    print(csv_first_row[0:12])
    round_list_val = [round(elem,4) for elem in ave_row_val[1:]]
    print(ave_row_val[0],round_list_val)
    round_list = [round(elem,4) for elem in ave_row[1:]]
    print(ave_row[0],round_list)
    # print(ave_row[0:12])
    return

def ignore_file(filename):
    global ignore_name
    for ignore in ignore_name:
        if filename.find(ignore) != -1:
            return True
    return False

def validation_file(filename):
    global validation_set
    for val in validation_set:
        if filename.find(val) != -1:
            return True
    return False

def csv_read():
    print("Reading directory:",input_dir)
    files = os.listdir(input_dir)
    files.sort(key=lambda x:x[:5],reverse = True)
    files.sort(key=lambda x:x[-8:])
    output_list = []
    output_list_val = []
    global Error_csv
    for filename in files:
        if ignore_file(filename):
            continue
        csv_path = os.path.join(input_dir,filename)
        f1_score_list = []
        with open(csv_path,newline='') as csv_file:
            rows = csv.reader(csv_file)
            TP_iou1 = []
            FP_iou1 = []
            FN_iou1 = []
            TP_iou2 = []
            FP_iou2 = []
            FN_iou2 = []
            first = True
            for row in rows:
                if first:
                    first = False
                    continue
                if str(row).find('nan') != -1 or str(row).find('inf') != -1:
                    continue
                TP_iou1.append(int(row[4]))
                FP_iou1.append(int(row[5]))
                FN_iou1.append(float(row[6]))
                TP_iou2.append(float(row[10]))
                FP_iou2.append(float(row[11]))
                FN_iou2.append(float(row[12]))
            split_filename = filename.split('_')
            # print(split_filename)
            f1_score_list.append(str(split_filename[2])+'_'+str(split_filename[3].split('.')[0])+'_'+str(split_filename[0])+'_'+str(split_filename[1]))
            f1_score_list.append(sum(TP_iou1))
            f1_score_list.append(sum(FP_iou1))
            f1_score_list.append(sum(FN_iou1))
            f1_score_list.append(sum(TP_iou2))
            f1_score_list.append(sum(FP_iou2))
            f1_score_list.append(sum(FN_iou2))
            if validation_file(filename):
                output_list_val.append(f1_score_list)
            else:
                output_list.append(f1_score_list)
    csv_write(output_list,output_list_val)
    return

if __name__ == '__main__':
    csv_read()