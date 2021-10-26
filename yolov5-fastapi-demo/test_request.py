import base64
import io
import json
import joblib
from PIL import Image
from numpy.lib.type_check import imag
import requests as r
from base64 import decodebytes

def send_request(file_list = [], 
					model_name = 'yolov5s',
					img_size = 640,
					download_image = False):

	#upload multiple files as list of tuples
	files = [('file_list', open(file,"rb")) for file in file_list]

	# check
	print(f'download image ? {download_image}')

	#pass the other form data here
	other_form_data = {'model_name': model_name,
					'img_size': img_size,
					'download_image': download_image}

	res = r.post("http://localhost:8000/detect/",
					data = other_form_data, 
					files = files)

	if download_image:
		json_data = res.json()

		for img_data in json_data:
			for bbox_data in img_data:
				#parse json to detect if the dict contains image data (base64) or bbox data
				if 'image_base64' in bbox_data.keys():
					#decode and show base64 encoded image
					img = Image.open(io.BytesIO(base64.b64decode(str(bbox_data['image_base64']))))
					img.show()
				else:
					#otherwise print json bbox data
					print(bbox_data)

	else:
		#if no images were downloaded, just display json response
		# print(json.loads(res.text))
		print(res.text)
		# with open(json.loads(res.text)) as f:
		# 	for line


response = send_request(file_list=["/home/kabin/web_konjac/dataset/konjac/images/IMG_8677.jpg"])
# , "/home/kabin/web_konjac/dataset/konjac/images/IMG_8678.jpg"

# if response is not None:
#     # parsed = json.loads(response)
#     print(json.dumps(response, indent=4, sort_keys=True))


loaded_rdf = joblib.load("../random_forest/rdf.joblib")
print(loaded_rdf)