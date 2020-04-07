import os
import csv
import urllib.request

def download_img(id, img_url):
	# header = {"Authorization": "Bearer " + api_token} # 设置http header
	header = {}
	request = urllib.request.Request(img_url, headers=header)
	try:
		response = urllib.request.urlopen(request)
		img_name = id+".png"
		filename = "../posters/"+ img_name
		if (response.getcode() == 200):
			with open(filename, "wb") as f:
				f.write(response.read()) # 将内容写入图片
			return filename
	except:
		print("error"+id)
		return "failed"

if __name__ == '__main__':
	# 下载要的图片
	api_token = "fklasjfljasdlkfjlasjflasjfljhasdljflsdjflkjsadljfljsda"
	
	row_names = ['movie_id', 'image_url']
	with open('movie_poster.csv', 'r', newline='') as in_csv:
		reader = csv.DictReader(in_csv, fieldnames=row_names, delimiter=',')
		for row in reader:
			movie_id = row['movie_id']
			image_url = row['image_url']
			if not os.path.exists('../posters/'+movie_id+'.png'):
				# download_img(movie_id, image_url, api_token)
				download_img(movie_id, image_url)