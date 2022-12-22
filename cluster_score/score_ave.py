import os
import csv
import numpy as np
from datetime import datetime

# set the output dir path
code_path = os.path.dirname(os.path.realpath(__file__))
# file_info = 'v26_dynamic_eps_MLP_class_v2_Nmin_parameter_with_feedback_and_without_tracking_filter_out_noises_dynamic_vel_threshold'
file_info = 'v30_dynamic_eps_MLP_class_v2_Nmin_parameter_with_feedback_and_without_tracking_filter_out_noises_dynamic_vel_threshold'

# current_files = os.listdir(code_path)
# current_files.sort(reverse = True, key=os.path.getmtime)
# current_files = list(filter(lambda x: '-' in x,current_files))
# read_fold_name = '2021-Nov-29'
# read_fold_name = current_files[0]   # 0:score_ave.py, 1: center.py, 2: Average folder
input_dir = os.path.join(code_path,'nuscenes','vmeasure_'+file_info)
# input_dir = os.path.join(code_path,'nuscenes','vmeasure_v3')
today = datetime.today().strftime('%Y-%b-%d')


ignore_name = []
# validataion log: 1(4), 3(7), 5(4), 10(19), 20(1), 23(11), 28(4), 40(6), 42(4), 43(10), 46(12), 50(10), 52(8), 56(12), 57(6), 58(9), 60(8), 65(15)
validation_set = \
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

output_dir = os.path.join(code_path,'nuscenes','Average')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

Error_csv = False
under_ave = False

def csv_write(output_list,output_list_val):
    global under_ave
    under_ave_count = 0
    csv_path = os.path.join(output_dir,today+'_Vmeasure_score.csv')
    # the first row of the csv file
    csv_first_row = ['File Name','Total Frames','Total Objects','Correct Objects','Over Objects','Under Objects','No Objects','V measure score','Homogeneity','H(C|K)','H(C)','Completeness','H(K|C)','H(K)','Ｓcene Num','Ave vel']
    ave_row =[]
    ave_row_val = []
    with open(csv_path,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_first_row)
        """
        Validation Data
        """
        if len(output_list_val)>0:
            ave_row_val.append('Ave'+' ('+str(len(output_list_val))+' bags)')
            output_val_array = np.asarray(output_list_val)
            output_val_array = np.asarray(output_val_array[:,1:],dtype=float)
            col_val_sum = output_val_array.sum(axis=0)
            ave_row_val.append(str(int(col_val_sum[0])))
            ave_row_val.append(int(col_val_sum[1]))
            col_val_sum = np.dot(output_val_array[:,0].reshape(1,len(output_val_array[:,0])),output_val_array[:,2:]) / int(col_val_sum[0])
            for i in range(len(col_val_sum[0])):
                ave_row_val.append(col_val_sum[0,i])
            writer.writerow(ave_row_val)
            
        """
        Training Data
        """
        if len(output_list)>0:
            ave_row.append('Ave'+' ('+str(len(output_list))+' bags)')
            output_array = np.asarray(output_list)
            output_array = np.asarray(output_array[:,1:],dtype=float)
            col_sum = output_array.sum(axis=0)
            ave_row.append(str(int(col_sum[0])))
            ave_row.append(int(col_sum[1]))
            col_sum = np.dot(output_array[:,0].reshape(1,len(output_array[:,0])),output_array[:,2:]) / int(col_sum[0])
            for i in range(len(col_sum[0])):
                ave_row.append(col_sum[0,i])
            writer.writerow(ave_row)
            
        """
        Write Data
        """
        if len(output_list_val)>0:
            for data in output_list_val:
                row = []
                for element in data:
                    row.append(element)
                writer.writerow(row)
        if len(output_list)>0:
            for data in output_list:
                row = []
                for element in data:
                    row.append(element)
                writer.writerow(row)

    print('\t',csv_first_row[0:8])
    print('Val\t',ave_row_val[0:8])
    print('Train\t',ave_row[0:8])
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
        csv_path = os.path.join(input_dir,filename)
        v_measure_score_list = []
        with open(csv_path,newline='') as csv_file:
            rows = csv.reader(csv_file)
            obj_num = []
            good_num = []
            multi_num = []
            under_mum = []
            no_num = []
            v_measure_score = []
            homo = []
            h_ck = []
            h_c = []
            comp = []
            h_kc = []
            h_k = []
            scene_num = []
            ave_vel =[]
            first = True
            for row in rows:
                if first:
                    first = False
                    continue
                if str(row[:]).find('nan') != -1 or str(row).find('inf') != -1:
                    continue
                obj_num.append(int(row[1]))
                good_num.append(int(row[2]))
                multi_num.append(int(row[3]))
                under_mum.append(int(row[4]))
                no_num.append(int(row[5]))
                v_measure_score.append(float(row[6]))
                homo.append(float(row[7]))
                h_ck.append(float(row[8]))
                h_c.append(float(row[9]))
                comp.append(float(row[10]))
                h_kc.append(float(row[11]))
                h_k.append(float(row[12]))
                scene_num.append(int(row[13]))
                ave_vel.append(float(row[14]))
            if len(v_measure_score) < 1:
                if not Error_csv:
                    Error_csv = True
                    print('\033[1;95m'+'Files Contain nan '+'\033[0m')
                print('\033[m'+str(filename)+'\033[0m')
                continue
            split_filename = filename.split('_')
            # v_measure_score_list.append(str(split_filename[2])+'_'+str(split_filename[4].split('.')[0])+'_'+str(split_filename[0])+'_'+str(split_filename[1]))
            v_measure_score_list.append(str(split_filename[2])+'_'+str(split_filename[4].split('.')[0]))
            v_measure_score_list.append(len(v_measure_score))
            v_measure_score_list.append(sum(obj_num))
            v_measure_score_list.append(sum(good_num)/sum(obj_num))
            v_measure_score_list.append(sum(multi_num)/sum(obj_num))
            v_measure_score_list.append(sum(under_mum)/sum(obj_num))
            v_measure_score_list.append(sum(no_num)/sum(obj_num))
            v_measure_score_list.append(sum(v_measure_score) / len(v_measure_score))
            v_measure_score_list.append(sum(homo) / len(homo))
            v_measure_score_list.append(sum(h_ck) / len(h_ck))
            v_measure_score_list.append(sum(h_c) / len(h_c))
            v_measure_score_list.append(sum(comp) / len(comp))
            v_measure_score_list.append(sum(h_kc) / len(h_kc))
            v_measure_score_list.append(sum(h_k) / len(h_k))
            v_measure_score_list.append(sum(scene_num) / len(scene_num))
            v_measure_score_list.append(sum(ave_vel) / len(ave_vel))
            if validation_file(filename):
                output_list_val.append(v_measure_score_list)
            else:
                output_list.append(v_measure_score_list)
    csv_write(output_list, output_list_val)
    return

if __name__ == '__main__':
    csv_read()