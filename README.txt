环境说明：
python	3.6.10
numpy	1.19.0
pandas	1.0.1
sklearn	0.22.1
cv2	3.3.1
scipy	1.4.1
tensorflow 2.2.0
dlib	19.6.1
imutils	0.5.3
pillow	7.2.0
xlrd	1.2.0
CUDA	None


1.Local_weighted_mean_register.py

用于裁剪以及配准人脸，包含类LWMRegister(standard_face, predictor_path, width=192, height=192, offset=24)

standard_face:	作为标准人脸的图像，不用经过裁剪
predictor_path:	dlib预训练人脸predictor所在地址
width, height:	目标图像的宽与高
offset:		为了避免映射超出图像边缘，在配准前给图像边缘留出的空余

使用方法：调用LWMRegister.run(face_seq, n = 6, amplify = 1.2, aligned=False)

face_seq: 	需要裁剪与配准的人脸图像序列
n:	计算LWM时采用的相邻点个数
amplify:	在配准时对图像进行一定比例的放大，避免映射超出图像边缘
aligned:	True表示face_seq已经经过裁剪、对齐，False表示没有经过


2.Eulerian_video_magnification.py

用于对视频（图像序列）进行动作放大，包含类EVM(fps=200, low=0.2, high=2.4, level=6, alpha=8, lam_c=16, iq_reduce=0.1)

fps:	视频的帧率
low, high:	动作放大的频率区间
level:	构建拉普拉斯金字塔的层数
alpha:	动作放大的倍数
lam_c:	参数，用于调节被放大的动作大小的上限deta
iq_reduce: 程序将图像转到YIQ颜色空间进行动作放大，iq_reduce表示对IQ通道的缩放倍数，缩小IQ可以减小噪声

使用方法: 调用EVM.run(img_seq)
img_seq: 被放大的图像序列


3.Temporal_interpolation_model.py

用于对图像序列进行时域插值，实现帧数的上采样或下采样，包含类TIM()
使用方法：调用TIM.run(image_seq, target_length)

image_seq:	需要被插值的图像序列
target_length:	目标帧数


4.Features_extraction.py

用于提取图像序列的特征，包括LBP-TOP、3DHOG、HOOF
使用方法：调用函数	get_ep_features(ep, uniform_dict = None, feature='LBP-TOP', t_times=4, y_times=4, x_times=4,
                    			x_radius = 1, y_radius = 1, t_radius = 4, xy_neighbor=8, xt_neighbor=8, yt_neighbor=8,
                    			xy_bins = 8, xt_bins = 12, yt_bins = 12,
                    			bins=8)

ep:		提取特征的图像序列
uniform_dict:	用于减小LBP-TOP特征维度，只有LBP-TOP特征需要
feature:		可选'LBP-TOP', '3DHOG', 'HOOF'
t_times, y_times, x_times:	将图像序列在t, y, x三个维度分成多少子块提取特征，推荐LBP-TOP的t_times=1
x_radius, y_radius, t_radius, xy_neighbor, xy_neighbor, yt_neighbor: 	LBP-TOP参数
xy_bins, xt_bins, yt_bins:	3DHOG参数
bins:		HOOF参数


5.Classification_and_evaluation.py

使用svm评估模型的分类效果，并输出最高的metric以及对应参数，使用方法：
调用函数get_best_average(data, labels, sub_list, kernel='linear', split='loso', average='macro')

data: 	特征数据
labels:	数据标签
sub_list:	每条数据属于哪个subject, 在程序中直接设置为df.Subject
kernel:	选择svm的核函数，可选'linear', 'poly', 'rbf'
split:	分割数据方式，可选'loso', '10-fold'
average:	计算Average F1-score的权重选择，可选'macro', 'micro',  'weighted'

查看某一特定参数下的分类效果，调用avg_score(data, label, sub_list, kernel='rbf', C=2, gamma=10, degree=3, decision_function_shape='ovr',
              					n_splits=10, split='loso', seed=7, average='macro')
data, label, sub_list, kernel, split, average:  同上
C, gamma, degree: 			svm参数
decision_function_shape: 		可选'ovo'与'ovr'，推荐'ovr'
n_splits:				k-fold验证的fold数
seed:				k-fold划分的随机数种子

6.main.py

将上述文件与main.py放在同一目录下，直接运行main.py：
a. 从同目录下的CASME II文件夹中提取数据，文件夹的结构为CASME II/subject_name/ep_name/image
b. 程序所需文件在CASME II文件夹下，分别为CASME2.xlsx, shape_predictor_68_face_landmarks.dat, UniformLBP8.txt
c. 程序将CASME II中第一个表情的第一张图片作为标准面部图像，对所有图像序列进行裁剪与配准，得到192*192的图像序列
d. 将配准后的结果存入result/lwm_result.npy中
e. 随后程序对图像序列进行动作放大，其中放大频率区间为[0.2Hz, 2.4Hz], 放大因子为8
f. 随后对图像序列进行时序插值，目标帧数为10帧
g. 随后对图像序列提取LBP-TOP、3DHOG、HOOF特征，存放于result/features/LBP_feature.npy (或HOG_feature.npy, HOOF_feature.npy)
h. 随后分别对特征使用svm进行分类，输出最好的分类结果与对应的svm参数

注意，main.py中参数并非最优，在实际使用时注意调整参数（包括TIM目标帧数、动作放大因子、特征提取分块数等）
