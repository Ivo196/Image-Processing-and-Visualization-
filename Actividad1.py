import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargar la imagen
imagen = cv2.imread("./image4.tif")

# Aplicar ajuste de brillo y contraste
# Alpha controla el contraste y beta controla el brillo
imagen_mejorada = cv2.convertScaleAbs(imagen, alpha=1, beta=30)


# Función para ajuste gamma
def ajuste_gamma(imagen, gamma=1.0):
    inv_gamma = 1.0 / gamma
    tabla = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(imagen, tabla)


# Aplicar ajuste gamma a la imagen
gamma = 1

imagen_ajustada_gamma = ajuste_gamma(imagen, gamma)

# Aplicar filtro Gaussiano para eliminar ruido
imagen_filtrada = cv2.GaussianBlur(imagen, (5, 5), 0)

# Ecualización del histograma
imagen_ecualizada = cv2.equalizeHist(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY))


#  Operaciones aritmeticas

# Suma de dos imágenes procesadas (por ejemplo, imagen mejorada y ecualizada)
imagen_combinada = cv2.addWeighted(imagen_ajustada_gamma, 0.5, imagen_mejorada, 0.5, 0)
####################################################################################
# Separar los canales de color
# canal_azul = imagen[:, :, 0]
# canal_verde = imagen[:, :, 1]
# canal_rojo = imagen[:, :, 2]

# # Disminuir la intensidad del canal azul
# canal_azul_disminuido = canal_azul * 1
# canal_verde_disminuido = canal_verde * 1
# canal_rojo_disminuido = canal_rojo * 1


# # Convertir los canales a tipo uint8 si es necesario
# canal_azul_disminuido = canal_azul_disminuido.astype(np.uint8)
# canal_rojo_disminuido = canal_rojo_disminuido.astype(np.uint8)
# canal_verde_disminuido = canal_verde_disminuido.astype(np.uint8)


# # Combinar los canales de nuevo en la imagen
# imagen_corregida = cv2.merge(
#     (canal_azul_disminuido, canal_verde_disminuido, canal_rojo_disminuido)
# )


def corregir_tinte_violeta(imagen):
    # Convertir la imagen a un espacio de color diferente para trabajar en los canales
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Ajustar el componente de matiz (Hue) para reducir el tinte violeta
    imagen_hsv[:, :, 0] = np.where(
        imagen_hsv[:, :, 0] >= 100, imagen_hsv[:, :, 0] - 100, imagen_hsv[:, :, 0] + 100
    )

    # Volver al espacio de color BGR
    imagen_corregida = cv2.cvtColor(imagen_hsv, cv2.COLOR_HSV2BGR)

    return imagen_corregida


# Cargar la imagen satelital con predominio de color violeta

imagen_original = imagen

imagen_corregida = corregir_tinte_violeta(imagen_original)

if imagen_original is not None:
    # Corregir el tinte violeta

    # Mostrar la imagen original y la imagen corregida
    cv2.imshow("Imagen Original", imagen_original)
    cv2.imshow("Imagen Corregida", imagen_corregida)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")


# Calcular el histograma
histogramaOriginal = cv2.calcHist([imagen], [0], None, [256], [0, 256])
histogramaNuevo = cv2.calcHist([imagen_mejorada], [0], None, [256], [0, 256])
histogramaNuevo2 = cv2.calcHist([imagen_ajustada_gamma], [0], None, [256], [0, 256])
histogramaNuevo3 = cv2.calcHist([imagen_filtrada], [0], None, [256], [0, 256])
histograma_ecualizado = cv2.calcHist([imagen_ecualizada], [0], None, [256], [0, 256])
histogramaNuevo4 = cv2.calcHist([imagen_combinada], [0], None, [256], [0, 256])


# Crear la figura y las subtramas
fig = plt.figure(figsize=(20, 15))

# Imagen Original y su histograma

plt.subplot(6, 6, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(6, 6, 7)
plt.plot(histogramaOriginal)
plt.title("Histograma Original")

# Imagen Mejorada y su histograma

plt.subplot(6, 6, 2)
plt.imshow(cv2.cvtColor(imagen_mejorada, cv2.COLOR_BGR2RGB))
plt.title("Imagen Mejorada")
plt.axis("off")

plt.subplot(6, 6, 8)
plt.plot(histogramaNuevo)
plt.title("Histograma Mejorado")

# Imagen con Ajuste Gamma y su histograma

plt.subplot(6, 6, 3)
plt.imshow(cv2.cvtColor(imagen_ajustada_gamma, cv2.COLOR_BGR2RGB))
plt.title("Imagen con Ajuste Gamma")
plt.axis("off")

plt.subplot(6, 6, 9)
plt.plot(histogramaNuevo2)
plt.title("Histograma Ajuste Gamma")

# Imagen con Filtrado(Gaussiano) y su histograma

plt.subplot(6, 6, 4)
plt.imshow(cv2.cvtColor(imagen_filtrada, cv2.COLOR_BGR2RGB))
plt.title("Imagen con Filtrado(Gaussiano)")
plt.axis("off")

plt.subplot(6, 6, 10)
plt.plot(histogramaNuevo3)
plt.title("Histograma con Filtrado(Gaussiano)")


# Imagen ecualizada y su histograma
plt.subplot(6, 6, 5)
plt.imshow(
    imagen_ecualizada,
)
plt.title("Imagen Ecualizada")
plt.axis("off")

plt.subplot(6, 6, 11)
plt.plot(histograma_ecualizado)
plt.title("Histograma I.Ecualizada")

# Imagen combinada y su histograma (Operacion aritmetica)
plt.subplot(6, 6, 6)
plt.imshow(imagen_corregida)
plt.title("Imagen Combinada(Aritmética)")
plt.axis("off")

plt.subplot(6, 6, 12)
plt.plot(histogramaNuevo4)
plt.title("Histograma I.Combinada")

# Ajustar el espacio entre las subtramas y mostrar la figura
plt.tight_layout()
plt.show()
