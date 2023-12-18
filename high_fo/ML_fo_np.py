
import load_data
import maf
import numpy as np
import matplotlib.pyplot as plt
import representations
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
import joblib


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LA_NORMAL = [1456,1481, 1505 ,1512 ,1516 ,1536 ,1539, 1541, 1563, 1566, 1568, 1576, 1583, 1603, 1613,
1617 ,1618 ,1626 ,1634 ,1642 ,1645 ,1646 ,1649 ,1652 ,1653 ,1661 ,1663, 1664,
1671 ,1676 ,1680 ,1687 ,1688 ,1689 ,1698 ,1700 ,1709 ,1714 ,1717 ,1720 ,1732,
1737 ,1741 ,1744 ,1746 ,1748 ,1749 ,1751 ,1753 ,1754 ,1760 ,1761 ,1763 ,1775 ,1776,
1778 ,1780 ,1783 ,1789 ,1795 ,1796 ,1797 ,1798 ,1805 ,1806 ,1807 ,1808 ,1817 ,1818,
1822 ,1823 ,1824 ,1828 ,1829 ,1833 ,1837 ,1841 ,1842 ,1846 ,1850 ,1851 ,1852,
1853 ,1854 ,1855 ,1858 ,1859 ,1860 ,1861 ,1862 ,1867 ,1868 ,1871 ,1874 ,1875 ,1877,
1878 ,1880 ,1883 ,1886 ,1889 ,1890 ,1891 ,1894 ,1895 ,1898 ,1899 ,1906 ,1907 ,1909,
1911 ,1912 ,1916 ,1918 ,1919 ,1920 ,1921 ,1923 ,1925 ,1928 ,1929 ,1930 ,1932 ,1938,
1939 ,1942 ,1946 ,1950 ,1951 ,1952 ,1957 ,1959 ,1962 ,1963 ,1964 ,1966 ,1967 ,1968,
1969 ,1970 ,1976 ,1979 ,1980 ,1981 ,1982 ,1983 ,1984 ,1986 ,1991 ,1993 ,1995 ,1999,
2000 ,2001 ,2002 ,2004 ,2006 ,2008 ,2010 ,2013 ,2016 ,2022 ,2023 ,2024 ,2027 ,2028,
2029 ,2030 ,2031]


 
LA_FIBROTIC =[1340 ,1360  ,1439 ,1447 ,1456 ,1457 ,1487 ,
 1563 ,1566 ,1568 ,1603 ,1613 ,1617 ,1618 ,1626 ,1631 ,1646 ,1649,
 1652 ,1653 ,1661 ,1663 ,1664 ,1676 ,1680 ,1687 ,1688 ,1689 ,1700 ,1709 ,1717,
 1732 ,1741 ,1744 ,1746 ,1748 ,1751 ,1754 ,1761 ,1778 ,1780 ,1783 ,1795 ,1796 ,1797,
 1806 ,1808 ,1818 ,1823 ,1828 ,1829 ,1837 ,1841 ,1850 ,1851 ,1855 ,1859,
 1860 ,1861 ,1867 ,1868 ,1875 ,1877 ,1878 ,1880 ,1886 ,1891 ,1894 ,1898 ,1906 ,1907,
 1912 ,1916 ,1918 ,1919 ,1921 ,1923 ,1928 ,1930 ,1932 ,1939 ,1951 ,1959,
 1963 ,1964 ,1968 ,1976 ,1979 ,1981 ,1984 ,1986 ,1999 ,2001 ,2002,
 2006 ,2024]


RA_NORMAL = [270  ,274  ,281  ,301  ,303  ,308 ,322  ,327  ,332,
  334  ,337  ,345  ,346  ,351  ,353  ,359  ,364  ,368  ,378  ,380  ,387  ,390,
  391  ,400  ,405  ,420  ,421  ,422  ,424  ,431  ,433  ,435  ,436  ,438,
  445  ,447  ,450  ,467  ,480  ,482  ,485  ,488  ,494  ,496  ,497  ,499  ,501,
  505  ,517  ,521  ,523  ,539  ,542  ,550  ,551  ,557  ,565  ,567  ,568  ,571,
  578  ,582  ,583  ,586  ,591  ,598  ,601  ,605  ,611,
  612  ,613  ,616  ,617  ,625  ,629  ,630  ,632  ,634  ,636  ,638  ,640,
  652  ,653  ,662  ,668  ,669  ,675  ,678  ,697  ,699  ,708  ,718  ,723  ,724,
  729  ,764  ,767  ,772  ,773  ,789  ,794  ,810  ,822  ,842,
  848  ,860  ,866  ,868 ]


def obtain_train_test():
    #Characteristics: fo and n_peaks

    DATASET = "valencia_sintetic_la_fibrotic"
    m_vm , heart_vertices, heart_faces,fs =  load_data.loadDataSet(DATASET)
    MAF_3D = maf.Maf()
    fo_map,peaks_map = MAF_3D.full_mesh_signal_proposal(m_vm, fs)

    fo_map = fo_map.tolist()
    peaks_map = (peaks_map ).tolist()


    #print(peaks_map)
    #print("_")
    #print(fo_map)

    x = list(zip(fo_map, peaks_map))

    #x = list(zip(fo_map, fo_map))

    labels = np.zeros((len(fo_map),1))
    labels[labels == 0] = False
    labels[LA_FIBROTIC] = True

    labels = labels.tolist()

    
    smote = SMOTE(random_state=42)
    x, labels = smote.fit_resample(x, labels)


    X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

    
    # Paso 4: Seleccionar un algoritmo de clasificación
    model = RandomForestClassifier(random_state=42)

    # Paso 5: Entrenar el modelo
    model.fit(X_train, y_train)

    # Paso 6: Hacer predicciones y evaluar el rendimiento
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    represent_model(model,fo_map,peaks_map, x,labels)

    joblib.dump(model, 'models/fo_np_model.joblib')

    return model


def predict_with_model(model, DATASET):

   
    m_vm , heart_vertices, heart_faces,fs =  load_data.loadDataSet(DATASET)
    MAF_3D = maf.Maf()
    fo_map,peaks_map = MAF_3D.full_mesh_signal_proposal(m_vm, fs)
    x = np.array([fo_map , peaks_map]).T

    predictions = model.predict(x)
    representations.moving_spectrum(heart_vertices, heart_faces,predictions.reshape(-1,1),np.array([0]))


def represent_model(model,feature1,feature2, x , labels):
   # Crear una malla de puntos para visualizar las regiones de decisión
    # Crear una malla de puntos para visualizar las regiones de decisión
    x = np.array(x)
    labels = np.array(labels)

    h = .001  # Tamaño del paso en la malla
    x_min, x_max = min([1]) , max([10])
    y_min, y_max = min([0]) , max([0.1])
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Realizar predicciones en la malla de puntos
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Visualizar las regiones de decisión sombreando las áreas correspondientes a cada clase
    plt.contourf(xx, yy, Z, alpha=0.3)

    # Visualizar los puntos de datos
    plt.scatter(x[:, 0][labels == 0], x[:, 1][labels == 0], c='red', marker='x', edgecolors='k', label='False')
    plt.scatter(x[:, 0][labels == 1], x[:, 1][labels == 1], c='blue', marker='.', s=100, edgecolors='k', label='True')

    #plt.scatter(x[0][labels],x[1][labels], cmap=plt.cm.coolwarm)

    # Etiquetas y título
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Regiones de Decisión del Clasificador Random Forest')
    plt.legend()

    # Mostrar la gráfica
    plt.show()

model = obtain_train_test()
predict_with_model(model , DATASET = "valencia_sintetic_la_fibrotic")
predict_with_model(model , DATASET = "valencia_sintetic_la_normal")
predict_with_model(model , DATASET = "valencia_sintetic_ra_normal")
