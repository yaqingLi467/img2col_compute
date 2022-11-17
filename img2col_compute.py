import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 自定义 feature map大小，kernel大小为3
out_c, n, in_c, h, w = 5, 1, 3, 5, 5  # 暂时先考虑正方形:13*13
im = torch.rand(n, in_c, h, w)
kernel_data = torch.rand(out_c, in_c, 3, 3)  # 后两位可以根据实际的kernel_size修改
kernel_size = kernel_data.shape[-1]  # 得到3，即kernel_size
kernel_pad = 1  # 可修改
kernel_stride = 1  # 可修改
print("input feature map shape is {}, kernel shape is {}".format(im.shape, kernel_data.shape))

# 使用torch自带的卷积，屏蔽掉bias
conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=kernel_stride, padding=kernel_pad,
                 padding_mode='zeros', bias=False)
conv.weight = nn.Parameter(kernel_data)
output = conv(im)
print("shape of output by torch.conv is {}".format(output.shape))

# 对feature map进行手动pad
padded_im = nn.functional.pad(im, (kernel_pad, kernel_pad, kernel_pad, kernel_pad), 'constant', 0)  # 会随着kernel_pad而变化
print("shape of padded feature map is {}".format(padded_im.shape))

# 预分配im2col的matrix: 大小为 [input_raw, input_col]
input_raw = int((((w+2*kernel_pad-kernel_size)/kernel_stride)+1) ** 2 * n)
input_col = in_c * kernel_size * kernel_size
im_col = torch.zeros(input_raw, input_col)
# 这里数据的不同channel左右连接了。行数是input_raw，即13^2*2。列数是左右链接3个channel形成的，这里是3*3*3。
print("多batch,多channel数据的im2映射:", im_col.shape)

# 截取每个3×3的子区域并塞入到im2col矩阵里面
padded_input_shape = padded_im.shape
k = 0
for idx_im in range(n):
    for i in range(1, padded_input_shape[-2] - 1):
        for j in range(1, padded_input_shape[-1] - 1):
            im_col[k, :] = padded_im[idx_im, :, i - 1:i + 2, j - 1:j + 2].clone().reshape(-1)
            k += 1  # 要随kernel_stride修改

# im2col和reshape后的kernel进行相乘
# reshape后的kernel大小为[kernel_size*kernel_size*in_c, out_c]
print("多个卷积核,每个核多channel，权重的im2映射:", kernel_data.reshape(kernel_size * kernel_size * in_c, out_c).shape)  # 权重的不同channel上下连接
output_mat = torch.matmul(im_col, kernel_data.reshape(kernel_size * kernel_size * in_c, out_c))
print("矩阵相乘的结果（col格式）：", output_mat.shape)  # [input_raw, out_c]

# 将结果reshape，这里面维度处理需要注意
output_mat_reshape = output_mat.permute(1, 0)  # 相当于转置，[out_c, input_raw]
output_mat_reshape = output_mat_reshape.reshape(out_c, n, h, w)  # [out_c, n,  h,w]
output_mat_reshape = output_mat_reshape.permute(1, 0, 2, 3)  # [ n, out_c, h,w]
print("矩阵相乘的结果（矩阵格式）：", output_mat_reshape.shape) # 可见img2col结果和pytorch卷积算子的结果一致

if __name__ == '__main__':
    # 假定存储器的大小为占据空间大小h*w
    print("————————————————————————————————————————————————————————————————————")
    print("when input feature map shape is {}，the required memory size of img2col is {} ".format(im.shape, im_col.shape))


