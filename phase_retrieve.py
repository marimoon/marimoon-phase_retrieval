import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

class PhaseRetrieve(object):
    def __init__(self, img_amp):
        # 周波数スペクトル画像（教示画像）
        self.img_amp = img_amp
        # 実空間画像に乱数画像を設定
        self.img_space = np.random.random(img_amp.shape)*255
        # 周波数画像にダミー画像を設定
        self.img_freq  = np.zeros(img_amp.shape, dtype=np.complex64)

        # 実空間の拘束変数
        self.beta = 0.95
        self.img_space_prev = self.img_space

        # shrink wrap
        self.img_mask = np.ones(img_amp.shape)
        self.interval_mask = 20
        self.th_mask       = 0.01
        self.gauss_sgm     = 5
        self.gauss_ratio   = 0.95
        self.mask_epoch    = 0

        # maskの初期設定
        self.init_mask()

        self.current_epoch = 0

    # 反復処理の実行(1サイクルずつ)
    def update(self):
        if self.current_epoch > 0:
            # 実空間の拘束条件
            self.constrain_space()

        self.keep_img_space()

        # 実空間画像をFourier変換
        self.img_freq = np.fft.fft2(self.img_space)

        # 周波数空間の拘束条件
        self.constrain_freq()

        # 周波数画像をFourier逆変換
        self.img_space = np.real(np.fft.ifft2(self.img_freq))

        # カウント
        self.current_epoch += 1

        # shrink_wrapの更新
        if (self.current_epoch%self.interval_mask == 0) and (self.mask_epoch < 5):
            self.update_mask()


    def init_mask(self):
        tmp = np.abs(np.fft.fftshift(np.fft.ifft2(self.img_amp * self.img_amp)))
        #tmp = np.abs(np.fft.fftshift(np.fft.ifft2(self.img_amp * self.img_amp)))
        tmp = gaussian_filter(tmp, sigma=self.gauss_sgm)
        max_ = np.max(tmp)
        self.img_mask = np.zeros(self.img_amp.shape)
        self.img_mask[ tmp >= max_ * self.th_mask ] = 1

        Image.fromarray(np.uint8(self.img_mask*255)).save("./debug/zz_mask_init.tif")


    # 実空間画像の保存
    def save_image(self, file):
        tmp = self.img_space
        tmp[ self.img_space < 0 ] = 0
        tmp[ self.img_space > 255 ] = 255
        Image.fromarray(np.uint8(np.abs(self.img_space))).save(file)

    # 実空間画像の制約(Hybrid Input Output法)
    def constrain_space(self):
        self.img_space[ self.img_mask < 1 ] = 0
        #self.img_space[ self.img_space < 0 ] = 0
        #self.img_space[ self.img_space < 0 ] = self.img_space[ self.img_space < 0 ]*0.1
        return
        # "mask内の正の画素"以外のindexを取得
        #index = ((self.img_mask > 0) * (self.img_space >= 0)) == False

        # 更新
        #self.img_space[index] = self.img_space_prev[index] - self.beta * self.img_space[index]
        #self.img_space[index] = self.img_space[index] - self.beta * self.img_space_prev[index]

    def keep_img_space(self):
        # 現在の実空間画像を保持
        self.img_space_prev = self.img_space

    # 周波数画像の制約(位相を維持したまま振幅を教示画像に更新)
    def constrain_freq(self):
        self.img_freq = self.img_freq / np.abs(self.img_freq) * self.img_amp

    # shrink_wrapの更新
    def update_mask(self):
        img_gauss = gaussian_filter(self.img_space, sigma=self.gauss_sgm)
        max_ = np.max(img_gauss)
        self.img_mask = np.zeros( self.img_amp.shape )
        self.img_mask[ img_gauss >= max_ * self.th_mask ] = 1
        Image.fromarray(np.uint8(self.img_mask*255)).save("./debug/zz_mask_iter{0:03d}.tif".format(self.current_epoch))

        self.gauss_sgm = self.gauss_sgm * self.gauss_ratio
        self.mask_epoch += 1

    # shrink_wrapの保存
    def save_mask(self, file):
        Image.fromarray(np.uint8(self.img_mask*255)).save(file)
