from PIL import Image

# Abre la imagen PNG
imagen_png = Image.open('ecg.png')



imagen_png = imagen_png.convert('RGB')
# Guarda la imagen en formato JPG
imagen_png.save('ecg.jpg', 'JPEG')

# Cierra la imagen PNG
imagen_png.close()