# To-See-Is-To-Touch
A vision-based tactile sensor, developed based on 9DTact.

先拍摄多张无接触无力图像求平均得到I0，
再对当前图像I1做差分，
把差分结果输入神经网络，
输出接触面的 3D 重建 + 6D wrench