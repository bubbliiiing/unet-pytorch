import os
import argparse
from tqdm import tqdm
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='datasets/SegmentationClass',
                        help='input path')
    parser.add_argument('--output_path', default='datasets/ReSize_SegmentationClass',
                        help='number of total epochs to run')
    config = parser.parse_args()

    return config

"""程序功能是对图片进行裁剪，把图片多余的部分进行裁剪，留下含有数据的部分"""

if __name__ == '__main__':

    config = vars(parse_args())
    imgs = os.listdir(config['input_path'])
    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])
    for img in tqdm(imgs):
        if img.endswith("jpg") or img.endswith("png"):
            left = 645
            top = 430
            right = 645 + 1024
            bottom = 430 + 1024
            #left = math.floor(width/2) - 512
            #right = math.floor(width/2) + 512
            #top = math.floor(height/2) - 512
            #bottom = math.floor(height/2) + 512
            try:
                im = Image.open(os.path.join(config['input_path'], img))
                width, height = im.size
                im1 = im.crop([left, top, right, bottom])
                cut_name = os.path.join(config['output_path'], img)
                im1.save(cut_name)
            except RuntimeError as e:
                print(e)
    print("转换完成!!")
