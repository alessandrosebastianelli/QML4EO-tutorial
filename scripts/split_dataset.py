import shutil
import random
import glob
import os

if __name__ == '__main__':


    root = '/content/dataset/training'
    split_factor = 0.2

    for cl in glob.glob(os.path.join(root, '*')):

        os.makedirs(cl.replace('training', 'validation'), exist_ok=True)

        imgs_in_cl = glob.glob(os.path.join(cl, '*'))
        random.shuffle(imgs_in_cl)

        validation_imgs = imgs_in_cl[0:int(len(imgs_in_cl)*split_factor)]

        for img in validation_imgs:
            shutil.move(img, img.replace('training', 'validation'))