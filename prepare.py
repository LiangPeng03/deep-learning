import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def prepare1():
    v=11
    cap = cv2.VideoCapture(f'data\\21.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 576, 576)

    success, frame = cap.read()
    #cv2.imshow('frame', frame)
    i = 0
    while success :
        if i % 5 == 0 :
            #frame = frame[0:144,90:200]
            cv2.imshow('frame', frame)
            frame = cv2.resize(frame, (144, 144), interpolation=cv2.INTER_LINEAR)  # 根据需要调整裁剪区域
            # 转换为灰度图像
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # 显示图像
            
            
            # 按不同的键保存到不同的文件夹
            key = cv2.waitKey(0)
            if key == ord('a'):
                print('a')
                cv2.imwrite(f'data\\open\\{v}_{i//5}.jpg', frame)
            elif key == ord('s'):
                print('s')
                cv2.imwrite(f'data\\middle\\{v}_{i//5}.jpg', frame)
            elif key == ord('d'):
                print('d')
                cv2.imwrite(f'data\\close\\{v}_{i//5}.jpg', frame)
            elif key == ord('w'):
                print('skip')

            elif key == ord('q'):
                break
        i += 1
        success, frame = cap.read()
    
    # 释放视频捕获对象和视频写入对象
    cap.release()
    cv2.destroyAllWindows()
    
def prepare2(folder_path, new_name_prefix):
     for count, filename in enumerate(os.listdir(folder_path)):
        # 构建旧文件路径和新文件路径
        old_file_path = os.path.join(folder_path, filename)
        # 只处理文件，跳过子文件夹
        if os.path.isfile(old_file_path):
            # 构建新的文件名
            new_file_name = f"{new_name_prefix}_{count}{os.path.splitext(filename)[1]}"#获取文件的扩展名
            new_file_path = os.path.join(folder_path, new_file_name)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} to {new_file_path}")

if __name__ == '__main__':
    prepare2('data\\close', 'c')