import cv2
from tunnel_simulation import simulate_tunnel_environment

img = cv2.imread("D:/tunnel_accident_ai/src/preprocessing/sample.jpg")
tunnel_img = simulate_tunnel_environment(img)

cv2.imshow("Original", img)
cv2.imshow("Tunnel", tunnel_img)
cv2.waitKey(0)
