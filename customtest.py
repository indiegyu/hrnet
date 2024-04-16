import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import tqdm
from model.locator import Crowd_locator
from misc.utils import *
from PIL import Image, ImageOps
import  cv2 
from collections import OrderedDict

# dataset = 'SHHA'
dataRoot = 'custom_testing_file/testing_images'
test_list = 'test.txt' # test.txt

GPU_ID = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
torch.backends.cudnn.benchmark = True

netName = 'HR_Net' # options: HR_Net,VGG16_FPN
# model_path = './exp/01-03_13-54_JHU_HR_Net/ep_305_F1_0.676_Pre_0.764_Rec_0.607_mae_74.5_mse_332.6.pth'
model_path = '../PretrainedModels/SHHA-HR-ep_905_F1_0.715_Pre_0.760_Rec_0.675_mae_112.3_mse_229.9.pth'

out_file_name = f"custom_testing_file/output_text/{netName}_{test_list}"

# 이미지 정규화(테스팅에서는 굳이 쓸 필요가 없음)
# if dataset == 'NWPU':
#     mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
# if dataset == 'SHHA':
#     mean_std = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
# if dataset == 'SHHB':
#     mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])   
# if dataset == 'QNRF':
#     mean_std = ([0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449])  
# if dataset == 'FDST':
#     mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])  
# if dataset == 'JHU':
#     mean_std = ([0.429683953524, 0.437104910612, 0.421978861094], [0.235549390316, 0.232568427920, 0.2355950474739]) 


# Mean STD 를 가지고 정규화를 적용한 코드 마찬가지로 테스팅에서는 굳이 쓸 필요가 없음
# img_transform = standard_transforms.Compose([
#         standard_transforms.ToTensor(),
#         standard_transforms.Normalize(*mean_std)
#     ])
# restore = standard_transforms.Compose([
#         own_transforms.DeNormalize(*mean_std),
#         standard_transforms.ToPILImage()
#     ])

import os


#이미지 파일의 이름을 받는 함수값 생성
def get_image_filenames(directory):
    image_filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 이미지 파일의 확장자를 제거하여 리스트에 추가합니다.
            image_filenames.append(filename.split('.')[0])  # .jpg 부분을 제거한 이름만 추가합니다.
    return image_filenames


# 디렉토리 안에 있는 이미지 파일 이름들 추출
image_names = get_image_filenames(dataRoot)

# 결과 출력(테스트용 코드)
# print("이미지 파일 이름들:")
# for name in image_names:
#     print(name)






def main():
    #이거 테스트 파일안에 있는 사진들의 목록을 적어놓은거라 그냥 무시하면 됨
    # txtpath = os.path.join(dataRoot, test_list)
    # #txtpath = dataRoot+'/'+test_list
    # with open(txtpath) as f:
    #     lines = f.readlines()                            
    test(image_names, model_path)


def get_boxInfo_from_Binar_map(Binar_numpy, min_area=3):
    Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
    assert Binar_numpy.ndim == 2
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)  # centriod (w,h)

    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]
    pre_data = {'num': len(points), 'points': points}
    return pre_data, boxes


def test(file_list, model_path):

    net = Crowd_locator(netName,GPU_ID,pretrained=True)
    net.cuda()
    state_dict = torch.load(model_path)
    if len(GPU_ID.split(','))>1:
        net.load_state_dict(state_dict)
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    net.eval()

    gts = []
    preds = []

    file_list = tqdm.tqdm(file_list)
    for infos in file_list:
        filename = infos.split()[0]
        #원본코드에서 images라는 파일을 참조하니까 파일경로를 맞춰주기 위해서 지워봄
        # imgname = os.path.join(dataRoot, 'images', filename + '.jpg')
        imgname = os.path.join(dataRoot, filename + '.jpg')
        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')
        #특정 이미지 정규화 코드라 일단 주석처리해봄
        # img = img_transform(img)[None, :, :, :]
        #특정 정규화 대신에 텐서로 변환하는 코드를 추가해줌
        img = standard_transforms.ToTensor()(img).unsqueeze(0).cuda()
        
        
        slice_h, slice_w = 512,1024
        slice_h, slice_w = slice_h, slice_w
        with torch.no_grad():
            img = Variable(img).cuda()
            b, c, h, w = img.shape
            crop_imgs, crop_dots, crop_masks = [], [], []
            if h * w < slice_h * 2 * slice_w * 2 and h % 16 == 0 and w % 16 == 0:
                [pred_threshold, pred_map, __] = [i.cpu() for i in net(img, mask_gt=None, mode='val')]
            else:
                if h % 16 != 0:
                    pad_dims = (0, 0, 0, 16 - h % 16)
                    h = (h // 16 + 1) * 16
                    img = F.pad(img, pad_dims, "constant")


                if w % 16 != 0:
                    pad_dims = (0, 16 - w % 16, 0, 0)
                    w = (w // 16 + 1) * 16
                    img = F.pad(img, pad_dims, "constant")


                for i in range(0, h, slice_h):
                    h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                    for j in range(0, w, slice_w):
                        w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                        crop_imgs.append(img[:, :, h_start:h_end, w_start:w_end])
                        mask = torch.zeros(1,1,img.size(2), img.size(3)).cpu()
                        mask[:, :, h_start:h_end, w_start:w_end].fill_(1.0)
                        crop_masks.append(mask)
                crop_imgs, crop_masks =  torch.cat(crop_imgs, dim=0), torch.cat(crop_masks, dim=0)

                # forward may need repeatng
                crop_preds, crop_thresholds = [], []
                nz, period = crop_imgs.size(0), 4
                for i in range(0, nz, period):
                    [crop_threshold, crop_pred, __] = [i.cpu() for i in net(crop_imgs[i:min(nz, i+period)],mask_gt = None, mode='val')]
                    crop_preds.append(crop_pred)
                    crop_thresholds.append(crop_threshold)

                crop_preds = torch.cat(crop_preds, dim=0)
                crop_thresholds = torch.cat(crop_thresholds, dim=0)

                # splice them to the original size
                idx = 0
                pred_map = torch.zeros(b, 1, h, w).cpu()
                pred_threshold = torch.zeros(b, 1, h, w).cpu().float()
                for i in range(0, h, slice_h):
                    h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                    for j in range(0, w, slice_w):
                        w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                        pred_map[:, :, h_start:h_end, w_start:w_end] += crop_preds[idx]
                        pred_threshold[:, :, h_start:h_end, w_start:w_end] += crop_thresholds[idx]
                        idx += 1
                mask = crop_masks.sum(dim=0)
                pred_map = (pred_map / mask)
                pred_threshold = (pred_threshold / mask)

            a = torch.ones_like(pred_map)
            b = torch.zeros_like(pred_map)
            binar_map = torch.where(pred_map >= pred_threshold, a, b)

            pred_data, boxes = get_boxInfo_from_Binar_map(binar_map.cpu().numpy())

            with open(out_file_name, 'a') as f:

                f.write(filename + ' ')
                f.write(str(pred_data['num']) + ' ')
                for ind,point in enumerate(pred_data['points'],1):
                    if ind < pred_data['num']:
                        f.write(str(int(point[0])) + ' ' + str(int(point[1])) + ' ')
                    else:
                            f.write(str(int(point[0])) + ' ' + str(int(point[1])))
                f.write('\n')
                f.close()
    visualizing()
        # record.close()


def visualizing():
    for filename in image_names:
        # 원본 이미지 불러오기
        imgname = os.path.join(dataRoot, filename + '.jpg')
        img = cv2.imread(imgname)
        # 예측된 위치 파일 불러오기
        pred_file_path = f"custom_testing_file/output_text/{netName}_{test_list}"
        with open(pred_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                data = line.split()
                #열갯수
                if data[0] == filename:
                    # number of object detected by the AI model(Ai가 감지한 객체 갯수)
                    num_points = int(data[1])

                    #객체의 x,y 좌표를 나타낸 리스트[x,y]형태로
                    points = [int(data[i]) for i in range(2, len(data))]
                    
                    # 예측된 위치에 원 그리기
                    for i in range(0, num_points * 2, 2):
                        x, y = points[i], points[i+1]
                        cv2.circle(img, (x, y), radius=10, color=(0, 255, 0), thickness=-1)
        
        # 시각화된 이미지 저장
        output_img_path = os.path.join("custom_testing_file/visualized_images" , f"_{num_points}_{filename}_visualized.jpg")
        cv2.imwrite(output_img_path, img)
        print(filename, "에서 감지된 객체는 총", num_points, "개 입니다.")
    print("예측된 위치를 시각화한 이미지를 저장했습니다.")



if __name__ == '__main__':
    main()


