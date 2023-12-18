import aspose.pdf as ap
import io
import os
import fitz  # Importa la biblioteca PyMuPDF


def pdf_2_jpg(FILE_NAME, DIR):

    # Ruta al archivo PDF que deseas convertir
    archivo_pdf = DIR + "/" + FILE_NAME

    print(archivo_pdf)
    # Abre el archivo PDF
    documento = fitz.open(archivo_pdf)

    # Itera a través de las páginas del PDF
    for pagina_numero in range(documento.page_count):
        # Obtiene la página actual
        pagina = documento.load_page(pagina_numero)
        
        # Convierte la página a una imagen PNG
        imagen = pagina.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        
        # Guarda la imagen en un archivo PNG
        imagen.save( archivo_pdf + f'pg_{pagina_numero + 1}.jpg')

    # Cierra el documento PDF
    documento.close()





def dir_pdf_2_jpg(DIR):

    # Ruta de la carpeta que deseas analizar
    carpeta = DIR

    # Lista para almacenar los nombres de los archivos PDF
    archivos_pdf = []

    # Recorre todos los archivos en la carpeta
    for nombre_archivo in os.listdir(carpeta):
        # Comprueba si el nombre del archivo termina en ".pdf"
        if nombre_archivo.endswith('.pdf'):
            # Si termina en ".pdf", añádelo a la lista
            archivos_pdf.append(nombre_archivo)

    # Imprime la lista de archivos PDF
    print("Archivos PDF en la carpeta:")
    for i in archivos_pdf:
        print(i)
        pdf_2_jpg(i, DIR)


dir_pdf_2_jpg("pdf_2_jpg")