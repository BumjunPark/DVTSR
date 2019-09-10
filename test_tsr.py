import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time
import imageio
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="PyTorch Robust Temporal Super Resolution for Dynamic Motion Video")
parser.add_argument("--cuda", action="store_true", help="Use cuda? Default: True")
parser.add_argument("--model0", default="F:/AIM2019/TSR/final_weight/weight0.pth", type=str, help="Model path0")
parser.add_argument("--model1", default="F:/AIM2019/TSR/final_weight/weight1.pth", type=str, help="Model path1")
parser.add_argument("--model2", default="F:/AIM2019/TSR/final_weight/weight2.pth", type=str, help="Model path2")
parser.add_argument("--data", default="F:/AIM2019/data/TSR_data/test/test_15fps", type=str, help="Test data path")
parser.add_argument("--gpu", default='0', help="GPU number to use when testing. Default: 0")
parser.add_argument("--result0", default="./result0/", type=str, help="Result path Default: ./result0/")
parser.add_argument("--result1", default="./result1/", type=str, help="Result path Default: ./result1/")
parser.add_argument("--result2", default="./result2/", type=str, help="Result path Default: ./result2/")

opt = parser.parse_args()

cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model0 = torch.load(opt.model0, map_location=lambda storage, loc: storage)["model"]
if opt.model1:
    model1 = torch.load(opt.model1, map_location=lambda storage, loc: storage)["model"]
if opt.model2:
    model2 = torch.load(opt.model2, map_location=lambda storage, loc: storage)["model"]

avg_elapsed_time = 0.0

folder_list = os.listdir(opt.data)

for folder in folder_list:
    file_list = os.listdir(opt.data + '/' + folder)
    for i in range(len(file_list)-1):

        file0 = file_list[i]
        file1 = file_list[i + 1]
        number = int(file0[:-4])
        name0 = folder + '_' + str(number + 2).zfill(8) + '.png'
        ens_name0 = folder + '_' + str(number + 2).zfill(8) + '_ens.png'
        name1 = folder + '_' + str(number + 4).zfill(8) + '.png'
        ens_name1 = folder + '_' + str(number + 4).zfill(8) + '_ens.png'
        name2 = folder + '_' + str(number + 6).zfill(8) + '.png'
        ens_name2 = folder + '_' + str(number + 6).zfill(8) + '_ens.png'

        input0 = imageio.imread(opt.data + '/' + folder + '/' + file0).transpose(2, 0, 1)
        input1 = imageio.imread(opt.data + '/' + folder + '/' + file1).transpose(2, 0, 1)

        [c, patch_h, patch_w] = input0.shape

        test_data0 = []
        test_data1 = []
        out_data = []
        temp01 = np.zeros((c, patch_w, patch_h))
        temp11 = np.zeros((c, patch_w, patch_h))
        temp02 = np.zeros((c, patch_h, patch_w))
        temp12 = np.zeros((c, patch_h, patch_w))
        temp03 = np.zeros((c, patch_w, patch_h))
        temp13 = np.zeros((c, patch_w, patch_h))
        ens_output = np.zeros((c, patch_h, patch_w))

        if opt.model1:
            out_data2 = []
            ens_output2 = np.zeros((c, patch_h, patch_w))

        if opt.model2:
            out_data3 = []
            ens_output3 = np.zeros((c, patch_h, patch_w))

        test_data0.append(input0)
        test_data1.append(input1)

        test_data0.append(np.fliplr(input0).copy())
        test_data1.append(np.fliplr(input1).copy())

        for a in range(c):
            temp01[a, :, :] = np.rot90(input0[a, :, :], 1)
            temp11[a, :, :] = np.rot90(input1[a, :, :], 1)
        test_data0.append(temp01)
        test_data1.append(temp11)

        test_data0.append(np.fliplr(test_data0[2]).copy())
        test_data1.append(np.fliplr(test_data1[2]).copy())

        for a in range(c):
            temp02[a, :, :] = np.rot90(input0[a, :, :], 2)
            temp12[a, :, :] = np.rot90(input1[a, :, :], 2)
        test_data0.append(temp02)
        test_data1.append(temp12)

        test_data0.append(np.fliplr(test_data0[4]).copy())
        test_data1.append(np.fliplr(test_data1[4]).copy())

        for a in range(c):
            temp03[a, :, :] = np.rot90(input0[a, :, :], 3)
            temp13[a, :, :] = np.rot90(input1[a, :, :], 3)
        test_data0.append(temp03)
        test_data1.append(temp13)

        test_data0.append(np.fliplr(test_data0[6]).copy())
        test_data1.append(np.fliplr(test_data1[6]).copy())

        for a in range(8):
            [tc, th, tw] = test_data0[a].shape
            data0 = Variable(torch.from_numpy(test_data0[a]/255.).float()).view(1, tc, th, tw)
            data1 = Variable(torch.from_numpy(test_data1[a]/255.).float()).view(1, tc, th, tw)

            if cuda:
                model0 = model0.cuda()
                data0 = data0.cuda()
                data1 = data1.cuda()

                if opt.model1:
                    model1 = model1.cuda()
                if opt.model2:
                    model2 = model2.cuda()

            start_time = time.time()
            with torch.no_grad():
                output = model0(data0, data1)
                out_data.append(output.cpu().detach().numpy().astype(np.float32))
                if opt.model1:
                    output2 = model1(data0, data1)
                    out_data2.append(output2.cpu().detach().numpy().astype(np.float32))
                if opt.model2:
                    output3 = model2(data0, data1)
                    out_data3.append(output3.cpu().detach().numpy().astype(np.float32))

            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

        results = np.zeros((8, c, patch_h, patch_w))

        results[0, :, :, :] = out_data[0][0, :, :, :]
        results[1, :, :, :] = np.fliplr(out_data[1][0, :, :, :])
        temp1 = np.fliplr(out_data[3][0, :, :, :]).copy()
        temp2 = np.fliplr(out_data[5][0, :, :, :]).copy()
        temp3 = np.fliplr(out_data[7][0, :, :, :]).copy()
        for a in range(c):
            results[2, a, :, :] = np.rot90(out_data[2][0, a, :, :], 3)
            results[3, a, :, :] = np.rot90(temp1[a, :, :], 3)
            results[4, a, :, :] = np.rot90(out_data[4][0, a, :, :], 2)
            results[5, a, :, :] = np.rot90(temp2[a, :, :], 2)
            results[6, a, :, :] = np.rot90(out_data[6][0, a, :, :], 1)
            results[7, a, :, :] = np.rot90(temp3[a, :, :], 1)

        if opt.model1:
            results2 = np.zeros((8, c, patch_h, patch_w))

            results2[0, :, :, :] = out_data2[0][0, :, :, :]
            results2[1, :, :, :] = np.fliplr(out_data2[1][0, :, :, :])
            temp1 = np.fliplr(out_data2[3][0, :, :, :]).copy()
            temp2 = np.fliplr(out_data2[5][0, :, :, :]).copy()
            temp3 = np.fliplr(out_data2[7][0, :, :, :]).copy()
            for a in range(c):
                results2[2, a, :, :] = np.rot90(out_data2[2][0, a, :, :], 3)
                results2[3, a, :, :] = np.rot90(temp1[a, :, :], 3)
                results2[4, a, :, :] = np.rot90(out_data2[4][0, a, :, :], 2)
                results2[5, a, :, :] = np.rot90(temp2[a, :, :], 2)
                results2[6, a, :, :] = np.rot90(out_data2[6][0, a, :, :], 1)
                results2[7, a, :, :] = np.rot90(temp3[a, :, :], 1)

        if opt.model2:
            results3 = np.zeros((8, c, patch_h, patch_w))

            results3[0, :, :, :] = out_data3[0][0, :, :, :]
            results3[1, :, :, :] = np.fliplr(out_data3[1][0, :, :, :])
            temp1 = np.fliplr(out_data3[3][0, :, :, :]).copy()
            temp2 = np.fliplr(out_data3[5][0, :, :, :]).copy()
            temp3 = np.fliplr(out_data3[7][0, :, :, :]).copy()
            for a in range(c):
                results3[2, a, :, :] = np.rot90(out_data3[2][0, a, :, :], 3)
                results3[3, a, :, :] = np.rot90(temp1[a, :, :], 3)
                results3[4, a, :, :] = np.rot90(out_data3[4][0, a, :, :], 2)
                results3[5, a, :, :] = np.rot90(temp2[a, :, :], 2)
                results3[6, a, :, :] = np.rot90(out_data3[6][0, a, :, :], 1)
                results3[7, a, :, :] = np.rot90(temp3[a, :, :], 1)

        for a in range(8):
            ens_output += results[a, :, :, :]
            if opt.model1:
                ens_output2 += results2[a, :, :, :]
            if opt.model2:
                ens_output3 += results3[a, :, :, :]

        out = results[0, :, :, :]
        out[out < 0] = 0
        out[out > 1] = 1.0
        ens_output /= 8.0
        ens_output[ens_output < 0] = 0
        ens_output[ens_output > 1] = 1.0
        if opt.model1:
            out2 = results2[0, :, :, :]
            out2[out2 < 0] = 0
            out2[out2 > 1] = 1.0
            ens_output2 /= 8.0
            ens_output2[ens_output2 < 0] = 0
            ens_output2[ens_output2 > 1] = 1.0
        if opt.model2:
            out3 = results3[0, :, :, :]
            out3[out3 < 0] = 0
            out3[out3 > 1] = 1.0
            ens_output3 /= 8.0
            ens_output3[ens_output3 < 0] = 0
            ens_output3[ens_output3 > 1] = 1.0

        out = out * 255.0
        out = np.uint8(np.round(out.transpose(1, 2, 0)))
        imageio.imwrite(opt.result0 + name0, out)
        ens_output = ens_output * 255.0
        ens_output = np.uint8(np.round(ens_output.transpose(1, 2, 0)))
        imageio.imwrite(opt.result0 + ens_name0, ens_output)
        if opt.model1:
            out2 = out2 * 255.0
            out2 = np.uint8(np.round(out2.transpose(1, 2, 0)))
            imageio.imwrite(opt.result1 + name1, out2)
            ens_output2 = ens_output2 * 255.0
            ens_output2 = np.uint8(np.round(ens_output2.transpose(1, 2, 0)))
            imageio.imwrite(opt.result1 + ens_name1, ens_output2)
        if opt.model2:
            out3 = out3 * 255.0
            out3 = np.uint8(np.round(out3.transpose(1, 2, 0)))
            imageio.imwrite(opt.result2 + name2, out3)
            ens_output3 = ens_output3 * 255.0
            ens_output3 = np.uint8(np.round(ens_output3.transpose(1, 2, 0)))
            imageio.imwrite(opt.result2 + ens_name2, ens_output3)

time_per_image = avg_elapsed_time / (len(folder_list) * (len(file_list) - 1) * 8)

if opt.model1:
    if opt.model2:
        time_per_image /= 3.0
    else:
        time_per_image /= 2.0

print("run time for a image: {}".format(time_per_image))


