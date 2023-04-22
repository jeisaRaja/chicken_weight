from PIL import Image


image = Image.open('tes_resize.jpg')
image.thumbnail((500,500), Image.LANCZOS)
image.save('tes2.jpg', "JPEG")