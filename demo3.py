from Stitcher import Stitcher
import cv2

# 读取拼接图片
imageA = cv2.imread("E:/Deeplearning/left.png")
imageB = cv2.imread("E:/Deeplearning/right.png")

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch((imageA, imageB), showMatches=True)

# 显示所有图片
cv2.imshow("E:/Deeplearning/left.png", imageA)
cv2.imshow("E:/Deeplearning/right.png", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()