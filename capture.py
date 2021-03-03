import pyautogui
from PIL import Image


class Capture():
    def split_img(self, img):
        # 読み込んだ画像を200*200のサイズで54枚に分割する
        height = 30
        width = 30
        split_size = 7

        buff = []
        # 縦の分割枚数
        for h1 in range(split_size):
            # 横の分割枚数
            for w1 in range(split_size):
                w2 = w1 * height
                h2 = h1 * width
                c = img.crop((w2, h2, width + w2, height + h2))
                yield c

    def taking_pictures(self):
        # pyautogui.screenshot(region=(1175, 669, 211, 211)).save('./capture/origin.png')
        pyautogui.screenshot(region=(535, 676, 211, 211)).save('./capture/origin.png')
        img = Image.open('./capture/origin.png')
        for j, im in enumerate(self.split_img(img)):
            im.save('./capture/_{:02}.png'.format(j))

if __name__ == '__main__':
    import time
    cap = Capture()
    time.sleep(3)
    cap.taking_pictures()

# region=(668, 533, 211, 211)