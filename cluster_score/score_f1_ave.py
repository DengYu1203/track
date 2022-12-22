import os
import csv
import numpy as np
from datetime import datetime

# set the output dir path
code_path = os.path.dirname(os.path.realpath(__file__))
dataset_name = ['radar_scenes','nuscenes']
score_dataset = dataset_name[0]

# ablation
file_info_list = [  'v75_ablation_eps_d_Nmin_d_v_d',
                    'v75_ablation_eps_d_Nmin_d_v_p1',
                    'v76_ablation_eps_d_Nmin_d_v_p2',
                    'v77_ablation_eps_d_Nmin_d_v_p3',
                    'v78_ablation_eps_d_Nmin_d_v_p4',
                    'v79_ablation_eps_d_Nmin_d_v_p5']
file_info = 'v81_ablation_eps_d_Nmin_4_v_p2'


input_dir = os.path.join(code_path,score_dataset,'f1score_'+file_info)
today = datetime.today().strftime('%Y-%b-%d')

output_dir = os.path.join(code_path,score_dataset,'Average')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

Error_csv = False
under_ave = False
ignore_name = []
validation_set = ['sequence_107','sequence_135','sequence_14','sequence_24','sequence_53','sequence_68','sequence_85',
                  'sequence_111','sequence_138','sequence_153','sequence_31','sequence_58','sequence_6','sequence_89',
                  'sequence_122','sequence_147','sequence_155','sequence_42','sequence_5','sequence_73','sequence_93',
                  'sequence_130','sequence_148','sequence_19','sequence_48','sequence_63','sequence_79','sequence_99']

nu_validation_set = \
    ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']

def csv_write(output_list,output_list_val):
    global under_ave
    under_ave_count = 0
    csv_path = os.path.join(output_dir,today+'_f1score_'+file_info+'.csv')
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
        if len(output_list_val)>0:
            ave_row_val.append('Val'+' ('+str(len(output_list_val))+' bags)')
            output_val_array = np.asarray(output_list_val)
            output_val_array = np.asarray(output_val_array[:,1:],dtype=float)
            col_val_sum = output_val_array.sum(axis=0)
            # f1 score(iou>=0.3), precision, recall
            ave_row_val.append(float(2*col_val_sum[0]/(2*col_val_sum[0]+col_val_sum[1]+col_val_sum[2]))*100)
            ave_row_val.append(float(col_val_sum[0]/(col_val_sum[0]+col_val_sum[1]))*100)
            ave_row_val.append(float(col_val_sum[0]/(col_val_sum[0]+col_val_sum[2]))*100)
            ave_row_val.append(int(col_val_sum[0]))
            ave_row_val.append(int(col_val_sum[1]))
            ave_row_val.append(int(col_val_sum[2]))
            # f1 score(iou>=0.5), precision, recall
            ave_row_val.append(float(2*col_val_sum[3]/(2*col_val_sum[3]+col_val_sum[4]+col_val_sum[5]))*100)
            ave_row_val.append(float(col_val_sum[3]/(col_val_sum[3]+col_val_sum[4]))*100)
            ave_row_val.append(float(col_val_sum[3]/(col_val_sum[3]+col_val_sum[5]))*100)
            ave_row_val.append(int(col_val_sum[3]))
            ave_row_val.append(int(col_val_sum[4]))
            ave_row_val.append(int(col_val_sum[5]))
            writer.writerow(ave_row_val)

        """
        Training Data
        """
        if len(output_list)>0:
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
        if len(output_list_val)>0:
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
        if len(output_list)>0:
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
    if len(output_list_val)>0:
        round_list_val = [round(elem,2) for elem in ave_row_val[1:]]
        print(ave_row_val[0],round_list_val)
    if len(output_list)>0:
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

def find_file(filename):
    certain_log = 'log43'
    if filename.find(certain_log) != -1:
        return False
    return True

def validation_file(filename):
    global validation_set
    split_filename = filename.split('_')
    search_file = split_filename[2]+'_'+split_filename[3].split('.')[0]
    check_validation_set = validation_set
    if score_dataset==dataset_name[1]:
        check_validation_set=nu_validation_set
        search_file = split_filename[4].split('.')[0]
    for val in check_validation_set:    
        if search_file == val:
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
        # if find_file(filename):
        #     continue
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
            if len(TP_iou1)==0:
                continue
            if score_dataset == dataset_name[0]:
                f1_score_list.append(str(split_filename[2])+'_'+str(split_filename[3].split('.')[0]))
            else:
                f1_score_list.append(str(split_filename[2])+'_'+str(split_filename[4].split('.')[0]))
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
    # for file_info_item in file_info_list:
    #     file_info = file_info_item
    #     input_dir = os.path.join(code_path,score_dataset,'f1score_'+file_info)
    #     csv_read()