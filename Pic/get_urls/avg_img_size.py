import os
from PIL import Image

w = 0
h = 0
count = 0

for img in os.listdir('../posters/'):
	im = Image.open('../posters/'+img)#返回一个Image对象
	w += im.size[0]
	h += im.size[1]
	count += 1

w /= count
h /= count

print('avg_w = '+str(w))
print('avg_h = '+str(h))