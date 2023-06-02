import tensorflow as tf
import numpy as np
import cv2
import train
import xlwt
import xlrd
import os

ckeckpoint_dir = './checkpoint'
test_pic_dir = "./duck2/"

def pic_init(num, pic):
    image_dir = test_pic_dir + str(num) + '/' + pic
    # print(image_dir)
    image = cv2.imread(image_dir)
    image = cv2.resize(image, (80, 324))
    b, g, r = cv2.split(image)
    predict_data = []
    predict_data.append(r)
    predict_data = np.array(predict_data)
    predict_data = predict_data.astype(np.float32)
    return predict_data

def real_data_init():
    workbook = xlrd.open_workbook('weight.xls')
    Data_sheet = workbook.sheets()[0]
    rowNum = Data_sheet.nrows
    colNum = Data_sheet.ncols
    xls_list = []
    for i in range(rowNum):
        rowlist = []
        for j in range(colNum):
            rowlist.append(Data_sheet.cell_value(i, j))
        xls_list.append(rowlist)
    real_data = []
    for i in range(1,51):
        c = int((i-1)/10)
        r = (i-1)%10 + 2
        real_data.append(float(xls_list[r][c]))
    return real_data

def main():
    style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')
    style1 = xlwt.easyxf(num_format_str='D-MMM-YY')
    wb = xlwt.Workbook()
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./checkpoint/variable-72000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    result = graph.get_tensor_by_name("y:0")
    real_data = real_data_init()
    for duck_num in range(1,51):
        dirs = os.listdir(test_pic_dir + str(duck_num))
        dirs = sorted(dirs)
        num = 1
        ws = wb.add_sheet(str(duck_num))
        ws.write(0, 0, '序号')
        ws.write(0, 1, '预测值')
        ws.write(0, 2, '真实值')
        ws.write(0, 3, '误差')
        for pic in dirs:
            test_pic = pic_init(duck_num, pic)
            feed_dict = {X: test_pic, keep_prob: 1}
            a = sess.run(result, feed_dict)
            b = a[0]
            pre_weight = b[0]
            ws.write(num,0,int(pic.split('.')[0]))
            ws.write(num,1,float(pre_weight))
            ws.write(num, 2, real_data[duck_num-1])
            loss = abs(float(pre_weight) - float(real_data[duck_num - 1]))
            ws.write(num, 3, loss)
            pre_weight = None
            num = num + 1
        print('finish ' + str(duck_num) +'/50!')
    wb.save('saveReport.xls')
    print('Save the report.xls')

if __name__ == '__main__':
    main()