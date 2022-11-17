# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''————————————————————test1:N=1 C=1,kernel size=3的实现————————————————————————————————'''
'''
# 生成 feature map和kernel
im = torch.rand(1, 1, 16, 16)
kernel_data = torch.rand(1, 1, 3, 3)  # batch,channel,height,weight

# 利用pytorch的卷积 生成输出
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1, padding_mode='zeros', bias=False)
conv.weight = nn.Parameter(kernel_data)
output = conv(im)

""" image to col  """

input_shape = im.shape
kernel_shape = conv.weight.data.shape
# 预分配im2col的matrix
im_col = torch.zeros(input_shape[-1]*input_shape[-2], kernel_shape[-1]*kernel_shape[-2])  # weight*height

padded_im = nn.functional.pad(im, (1, 1, 1, 1), 'constant', 0)
padded_input_shape = padded_im.shape
# 截取每个3×3的子区域并塞入到im_col 里面
k = 0
for i in range(1, padded_input_shape[-2]-1):
    for j in range(1, padded_input_shape[-1]-1):
        im_col[k, :] = padded_im[0, 0, i-1:i+2, j-1:j+2].clone().reshape(-1)
        k += 1

# mat_col × kernel 并会reshape回原尺度。注意这里是矩阵乘法
output_mat = torch.matmul(im_col, kernel_data.view(9, 1))
output_mat_reshape = output_mat.reshape(1, 1, 16, 16)
'''

'''——————————————test2 自定义的N/C，但是kernel size还是坚持为3，可修改————————————————'''
# 自定义 feature map大小，当然，kernel大小还是3
out_c, n, in_c, h, w = 5, 2, 3, 13, 16
im = torch.rand(n, in_c, h, w)
kernel_data = torch.rand(out_c, in_c, 3, 3)  # 后两位可以根据实际的kernel_size修改
kernel_size = kernel_data.shape[-1]  # 得到3，即kernel_size
print("input feature map shape is {}, kernel shape is {}".format(im.shape, kernel_data.shape))

# 使用torch自带的卷积，屏蔽掉bias
conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=1, padding=1, padding_mode='zeros',
                 bias=False)
conv.weight = nn.Parameter(kernel_data)
output = conv(im)

print("shape of output by torch.conv is {}".format(output.shape))

# 对feature map进行手动pad
padded_im = nn.functional.pad(im, (1, 1, 1, 1), 'constant', 0)
print("shape of padded feature map is {}".format(padded_im.shape))

# 预分配im2col的matrix:
# 大小为 [n * h * w, in_c * kernel_size * kernel_size]
im_col = torch.zeros(n * w * h, in_c * kernel_size * kernel_size)
# 这里数据的不同channel左右连接了。行数因为手动pad,所以是n*(w*h)，这里是2*16*13。列数是左右链接3个channel形成的，这里是3*3*3。
print("多batch,多channel数据的im2映射:", im_col.shape)

# 截取每个3×3的子区域并塞入到im2col矩阵里面
padded_input_shape = padded_im.shape
k = 0
for idx_im in range(n):
    for i in range(1, padded_input_shape[-2] - 1):
        for j in range(1, padded_input_shape[-1] - 1):
            im_col[k, :] = padded_im[idx_im, :, i - 1:i + 2, j - 1:j + 2].clone().reshape(-1)
            k += 1

# im2col和reshape后的kernel进行相乘
# reshape后的kernel大小为[kernel_size*kernel_size*in_c, out_c]
print("多个卷积核,每个核多channel，权重的im2映射:", kernel_data.reshape(kernel_size * kernel_size * in_c, out_c).shape)  # 权重的不同channel上下连接
output_mat = torch.matmul(im_col, kernel_data.reshape(kernel_size * kernel_size * in_c, out_c))
print("矩阵相乘的结果（col格式）：", output_mat.shape)  # [n * w * h, out_c]

# 将结果reshape，这里面维度处理需要注意
output_mat_reshape = output_mat.permute(1, 0)  # [out_c, n * h * w]
output_mat_reshape = output_mat_reshape.reshape(out_c, n, h, w)  # [out_c, n,  h,w]
output_mat_reshape = output_mat_reshape.permute(1, 0, 2, 3)  # [ n, out_c, h,w]
print("矩阵相乘的结果（矩阵格式）：", output_mat_reshape.shape) # 可见img2col结果和pytorch卷积算子的结果一致

if __name__ == '__main__':
    print('PyCharm')

