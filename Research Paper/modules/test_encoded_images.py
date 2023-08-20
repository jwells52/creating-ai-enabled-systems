import base64
import sys
import io

from PIL import Image

if __name__ == '__main__':
    
    encoded_file_path = sys.argv[1]

    f = open(encoded_file_path, "r")
    im_b64 = f.read()
    print(im_b64)
    # # with open(encoded_file_path)

    # # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))
    img = img.save("test.jpg")