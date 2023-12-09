import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargar la imagen
imagen = cv2.imread("../image1.png")

# Aplicar ajuste de brillo y contraste
# Alpha controla el contraste y beta controla el brillo
imagen_mejorada = cv2.convertScaleAbs(imagen, alpha=2, beta=30)


# Función para ajuste gamma
def ajuste_gamma(imagen, gamma=1.0):
    inv_gamma = 1.0 / gamma
    tabla = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(imagen, tabla)


# Aplicar ajuste gamma a la imagen
gamma = 1.5
imagen_ajustada_gamma = ajuste_gamma(imagen, gamma)

# Aplicar filtro Gaussiano para eliminar ruido
imagen_filtrada = cv2.GaussianBlur(imagen, (5, 5), 0)

# Calcular el histograma
histogramaOriginal = cv2.calcHist([imagen], [0], None, [256], [0, 256])
histogramaNuevo = cv2.calcHist([imagen_mejorada], [0], None, [256], [0, 256])
histogramaNuevo2 = cv2.calcHist([imagen_ajustada_gamma], [0], None, [256], [0, 256])
histogramaNuevo3 = cv2.calcHist([imagen_filtrada], [0], None, [256], [0, 256])

# Crear la figura y las subtramas
fig = plt.figure(figsize=(15, 10))

# Imagen Original y su histograma

plt.subplot(4, 4, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(4, 4, 5)
plt.plot(histogramaOriginal)
plt.title("Histograma Original")

# Imagen Mejorada y su histograma

plt.subplot(4, 4, 2)
plt.imshow(cv2.cvtColor(imagen_mejorada, cv2.COLOR_BGR2RGB))
plt.title("Imagen Mejorada")
plt.axis("off")

plt.subplot(4, 4, 6)
plt.plot(histogramaNuevo)
plt.title("Histograma Mejorado")

# Imagen con Ajuste Gamma y su histograma

plt.subplot(4, 4, 3)
plt.imshow(cv2.cvtColor(imagen_ajustada_gamma, cv2.COLOR_BGR2RGB))
plt.title("Imagen con Ajuste Gamma")
plt.axis("off")

plt.subplot(4, 4, 7)
plt.plot(histogramaNuevo2)
plt.title("Histograma Ajuste Gamma")

# Imagen con Filtrado(Gaussiano) y su histograma

plt.subplot(4, 4, 4)
plt.imshow(cv2.cvtColor(imagen_filtrada, cv2.COLOR_BGR2RGB))
plt.title("Imagen con Filtrado(Gaussiano)")
plt.axis("off")

plt.subplot(4, 4, 8)
plt.plot(histogramaNuevo3)
plt.title("Histograma con Filtrado(Gaussiano)")


# Ajustar el espacio entre las subtramas y mostrar la figura
plt.tight_layout()
plt.show()
