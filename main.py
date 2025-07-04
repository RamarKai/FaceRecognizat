import sensor
import image
import lcd
import KPU as kpu
import time
from Maix import FPIOA, GPIO
import gc
from fpioa_manager import fm
from board import board_info
import utime
import uos

###################################################################################
'''
函数说明:
    该函数用于加载模型固件
'''
task_fd = kpu.load(0x300000)                                        #从flash加载固件
task_ld = kpu.load(0x400000)
task_fe = kpu.load(0x500000)

#task_fd = kpu.load("/sd/FaceDetection.smodel")                     #从SD加载固件
#task_ld = kpu.load("/sd/FaceLandmarkDetection.smodel")
#task_fe = kpu.load("/sd/FeatureExtraction.smodel")
###################################################################################

#fm.register(6, fm.fpioa.UART1_RX, force=True)                      #映射串口引脚
#fm.register(7, fm.fpioa.UART1_TX, force=True)

clock = time.clock()
fm.register(board_info.BOOT_KEY, fm.fpioa.GPIOHS0)                  #定义BOOOT按键GPIO
key_gpio = GPIO(GPIO.GPIOHS0, GPIO.IN)
start_processing = False

BOUNCE_PROTECTION = 50 #按键消抖
def set_key_state(*_):
    global start_processing
    start_processing = True
    utime.sleep_ms(BOUNCE_PROTECTION)

key_gpio.irq(set_key_state, GPIO.IRQ_RISING, GPIO.WAKEUP_NOT_SUPPORT)#中断

lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_hmirror(1)
sensor.set_vflip(0)
sensor.run(1)

anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437,6.92275, 6.718375, 9.01025)  # anchor for face detect
dst_point = [(44, 59), (84, 59), (64, 82), (47, 105),(81, 105)]  # standard face key point position
a = kpu.init_yolo2(task_fd, 0.5, 0.3, 5, anchor)
img_lcd = image.Image()
img_face = image.Image(size=(128, 128))
a = img_face.pix_to_ai()

record_ftr = []
record_ftrs = []
names = ['QKX', 'CXY', '3', '4', '5','6', '7', '8', '9', '10']      #按需修改标注
ACCURACY = 75                                                       #设置识别阈值百分比

####################################################################################
'''
函数魔改说明:
    Step1.注释read_feature(Addr),启用save_feature_sd(Addr)
    Step2.修改save_feature_sd(Addr)中Addr后缀
    Step3.按下BOOT将所捕获人脸特征向量保存至SD卡
    Step4.启用read_feature(Addr),注释save_feature_sd(Addr)
    Step5.按需添加修改read_feature(Addr),读取存储在SD中的特征向量数据
    Step6.按下RESET重启即可食用
    Attention.读取文件名需与保存文件名一致
'''
def read_feature(Addr):
    sd_path = Addr                          #之前保存特征向量的 SD 路径
    sd_file = open(sd_path, "rb")           #读取存储在 SD 中的特征向量数据
    record_ftr = bytearray(sd_file.read())  #赋值
    sd_file.close()                         #关闭文件
    record_ftrs.append(record_ftr)          #堆栈特征向量,把特征向量存于record_ftrs

def save_feature_sd(Addr):
    feature_file = open(Addr, "wb")         # 在SD卡上创建一个文件用于保存特征向量
    feature_file.write(bytearray(feature))  # 将特征向量转换为bytes并写入文件
    feature_file.close()
                                                                 #by_zyb_2023_07_14
###################################################################################

read_feature("/sd/featureQKX.bin")#Addr1
read_feature("/sd/featureCXY.bin")#Addr2

while (1):
    img = sensor.snapshot()
    clock.tick()
    code = kpu.run_yolo2(task_fd, img)
    if code:
        for i in code:
            # Cut face and resize to 128x128
            a = img.draw_rectangle(i.rect())
            face_cut = img.cut(i.x(), i.y(), i.w(), i.h())
            face_cut_128 = face_cut.resize(128, 128)
            a = face_cut_128.pix_to_ai()

            # a = img.draw_image(face_cut_128, (0,0))
            # Landmark for face 5 points
            fmap = kpu.forward(task_ld, face_cut_128)
            plist = fmap[:]
            le = (i.x() + int(plist[0] * i.w() - 10), i.y() + int(plist[1] * i.h()))
            re = (i.x() + int(plist[2] * i.w()), i.y() + int(plist[3] * i.h()))
            nose = (i.x() + int(plist[4] * i.w()), i.y() + int(plist[5] * i.h()))
            lm = (i.x() + int(plist[6] * i.w()), i.y() + int(plist[7] * i.h()))
            rm = (i.x() + int(plist[8] * i.w()), i.y() + int(plist[9] * i.h()))
            a = img.draw_circle(le[0], le[1], 4)
            a = img.draw_circle(re[0], re[1], 4)
            a = img.draw_circle(nose[0], nose[1], 4)
            a = img.draw_circle(lm[0], lm[1], 4)
            a = img.draw_circle(rm[0], rm[1], 4)

            # align face to standard position
            src_point = [le, re, nose, lm, rm]
            T = image.get_affine_transform(src_point, dst_point)
            a = image.warp_affine_ai(img, img_face, T)
            a = img_face.ai_to_pix()

            # a = img.draw_image(img_face, (128,0))
            del (face_cut_128)

            # calculate face feature vector
            fmap = kpu.forward(task_fe, img_face)
            feature = kpu.face_encode(fmap[:])
            reg_flag = False
            scores = []
            for j in range(len(record_ftrs)):
                score = kpu.face_compare(record_ftrs[j], feature)
                scores.append(score)
            max_score = 0
            index = 0
            for k in range(len(scores)):
                if max_score < scores[k]:
                    max_score = scores[k]
                    index = k
            if max_score > ACCURACY:
                a = img.draw_string(i.x(), i.y(), ("%s :%2.1f" % (
                    names[index], max_score)), color=(0, 255, 0), scale=2)
            else:
                a = img.draw_string(i.x(), i.y(), ("X :%2.1f" % (
                    max_score)), color=(255, 0, 0), scale=2)
            if start_processing:
                #save_feature_sd("/sd/featureCXY.bin")
                record_ftr = feature
                record_ftrs.append(record_ftr)
                start_processing = False
            break

    fps = clock.fps()
    print("%2.1f fps" % fps)
    a = lcd.display(img)
    gc.collect()
