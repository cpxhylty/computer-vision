from Stitcher import Stitcher
import cv2 # version 3.4.2.16 (and opencv-contrib-python)

# 读取拼接图片
imageA = cv2.imread("./left.jpg")
imageB = cv2.imread("./right.jpg")

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch((imageA, imageB), showMatches=True)

# 显示所有图片
'''
cv2.imshow("./left.jpg", imageA)
cv2.imshow("./right.jpg", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
'''
cv2.imwrite('./key points.jpg', vis)
cv2.imwrite('./stitched picture.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()