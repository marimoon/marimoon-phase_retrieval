import numpy as np
from PIL import Image

from phase_retrieve import PhaseRetrieve

def save_speckle(img, file):
    tmp = np.fft.fftshift(img)
    tmp = np.log(tmp+1.0)
    max_ = np.max(tmp)
    min_ = np.min(tmp)
    tmp  = (tmp - min_) / (max_ - min_) * 255
    Image.fromarray(np.uint8( tmp )).save(file)

def main(file):
    # 正解画像の読み込み
    with Image.open(file, "r") as buf:
        img = np.array(buf)
    Image.fromarray(np.uint8(img)).save("zz_extact.tif")

    # 周波数スペクトルの取得
    img_amp = np.abs( np.fft.fft2(img) )
    save_speckle(img_amp, "zz_speckle.tif")

    # 位相復元アプリの初期化
    application = PhaseRetrieve( img_amp )

    application.save_image("./debug/zz_image_iter000.tif")
    application.save_mask("./debug/zz_mask_iter000.tif")

    # 復元処理の実行
    for i in range(300):
        application.update()
        application.save_image("./debug/zz_image_iter{0:03d}.tif".format(i+1))

    application.save_image("zz_final.tif")

if __name__ == "__main__":
    file = "fruits_basket.tif"
    main(file)
