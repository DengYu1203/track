import numpy as np

center_list = []

def get_pt():
    pt_num = int(input('Total points needed to caluclate:'))
    pt_list = []
    for i in range(pt_num):
        p1 = input('Enter the coordinate of the pt'+str(i+1)+':')
        p1 = p1.replace('(','')
        p1 = p1.replace(')','')
        p1 = p1.split(',')
        np_p1 = np.array([float(p1[0]),float(p1[1])])
        print('p'+str(i+1)+':',np_p1)
        pt_list.append(np_p1)
    # p2 = input('Enter the coordinate of the pt2:')
    # p2 = p2.replace('(','')
    # p2 = p2.replace(')','')
    # p2 = p2.split(',')
    # np_p2 = np.array([float(p2[0]),float(p2[1])])
    # print('p2:',np_p2)
    pt_array = np.array(pt_list)
    pt = pt_array.sum(axis=0)  / pt_num
    return pt

def center_distance():
    global center_list
    # dis_mat = np.zeros((len(center_list),len(center_list)))
    np.set_printoptions(formatter={"float_kind": lambda x: "%0.3f" % x})
    dis_mat = np.zeros((len(center_list)+1,len(center_list)+1), dtype=object)
    dis_mat[0,0] = None
    for i in range(len(center_list)):
        dis_mat[i+1,0] = 'p'+str(i+1)
        dis_mat[0,i+1] = 'p'+str(i+1)
        for j in range(i+1,len(center_list)):
            dis_mat[i+1,j+1] = ((center_list[i][0] - center_list[j][0])**2 + (center_list[i][1] - center_list[j][1])**2)**0.5
    print(dis_mat,sep='\t')
    return

def main():
    pt = get_pt()
    global center_list
    center_list.append(pt)
    print('\ncenter:',pt)
    return

if __name__ == '__main__':
    try:
        while(True):
            main()
            print('')
    except KeyboardInterrupt:
        print("\n")
        center_distance()
        print("End the program due to KeyboardInterrupt")
    