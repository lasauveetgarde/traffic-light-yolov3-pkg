import cv2

index = 6.2
str_index = str(index)

print(type(str_index))
new_index = str_index.replace(".", "_", 1)
new_index=new_index+".png"
print(new_index)

img = cv2.imread(new_index, cv2.IMREAD_COLOR)
img = cv2.resize(img, (400, 400))
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
