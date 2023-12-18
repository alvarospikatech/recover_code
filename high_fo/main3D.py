import maf
import numpy as np
import load_data
import load_data
import matplotlib.pyplot as plt
import representations
import itertools
import spatial_analysis
from collections import Counter



#m_vm, heart_faces,heart_vertices, m_ve ,torso_faces,torso_vertices , torso_faces_cl,torso_vertices_cl = load_data.loadDataSet("utah_2018_sinus")


def spectrograma():

 
    DATASET = "valencia_sintetic_la_fibrotic"
    m_vm , heart_vertices, heart_faces,fs =  load_data.loadDataSet(DATASET)    
    t =  np.arange(0, m_vm.shape[1]/fs, 1/fs) #np.linspace(start=0,stop=dur, num=int(dur * Fs) ) #t = 0:ts:dur-ts; % Segundos

    potentials_modelograma,f_axis,t_axis = representations.calculate_modelogram_matrix(m_vm,fs,t)

    print(potentials_modelograma.shape)
    print(heart_vertices.shape)
    print(f_axis.shape)
    print(t_axis.shape)
    #representations.spectrogram_3D(heart_vertices, heart_faces, potentials_modelograma, f_axis, t_axis)
    
    potentials_spectrogram,f_axis,t_axis = representations.calculate_spectrogram_matrix(m_vm,fs)
    representations.spectrogram_3D(heart_vertices, heart_faces, potentials_spectrogram, f_axis, t_axis)

    5/0
    load = np.load('resultados/'+DATASET+".npy", allow_pickle=True).item()
    FAM_timeSignal = np.array(load['FAM_timeSignal'])
    potentials_spectrogram,f_axis,t_axis = representations.calculate_spectrogram_matrix(FAM_timeSignal,fs)
    representations.spectrogram_3D(heart_vertices, heart_faces, potentials_spectrogram, f_axis, t_axis)


def load_pipeline(DATASET ="valencia_sintetic_la_fibrotic" ,threshold = 7.1):


    
    m_vm , heart_vertices, heart_faces,fs =  load_data.loadDataSet(DATASET)
    t =  np.arange(0, m_vm.shape[1]/fs, 1/fs)

    load = np.load('resultados/'+DATASET+".npy", allow_pickle=True).item()
    f_axis = np.array([0])
    spectrum = np.array(load['fo_map'])
    #spectrum = spectrum /(np.max(spectrum))

    """
    f_axis,spectrum = representations.calculate_spectrum_matrix(t, m_vm)
    spectrum = np.abs(spectrum)
    spectrum = spectrum / np.max(spectrum)

    print(spectrum.shape)
    print(f_axis.shape)
    dif = np.abs(f_axis - 12)
    index = np.argmin(dif)
    spectrum = spectrum[:, 0:index]
    f_axis = f_axis[0:index]
    """

    #representations.moving_spectrum(heart_vertices, heart_faces,spectrum,f_axis)
    #oct_signal, oct_f_axis = spatial_analysis.third_octaves_spectrum(spectrum ,f_axis)
    print("_____caca______")
  
    automateFreq = spectrum[~np.isnan(spectrum)]
    automateFreq = automateFreq[automateFreq != 0]
    #automateFreq = np.floor(np.array(automateFreq))

    """
    print(automateFreq)
    plt.hist(automateFreq)
    plt.show()
    """
    most_common_value = max(automateFreq.tolist(), key=automateFreq.tolist().count)
    thressup= most_common_value*1.75
    if thressup >= 7: 
        thressup = 7

    if thressup <=4.5:
        thressup = 5.5

    print(thressup)

    if threshold == 0:
        print("holaa")
        threshold_new = thressup
    else:
        threshold_new = threshold

    spectrum[spectrum <=threshold_new ] = 0
    #spectrum[spectrum >12] = 0
    representations.moving_spectrum(heart_vertices, heart_faces,spectrum,f_axis)
    label_x, label_map, labels = spatial_analysis.connected_label_3d(spectrum,heart_vertices,heart_faces)
    label_x, label_map, labels = spatial_analysis.group_labels_by_distance(heart_vertices,labels,5)
    representations.moving_spectrum(heart_vertices, heart_faces, label_map,label_x)
    label_x, label_map, labels = spatial_analysis.labels_size_filtering(heart_vertices,labels,5)
    #representations.moving_spectrum(heart_vertices, heart_faces, label_map,label_x)
    if label_map.shape[1] != 0:
        label_map = np.where(label_map != 0, 1, 0)
        label_map = np.sum(label_map, axis=1).reshape(-1, 1)
        label_axis = np.array([0])

    representations.moving_spectrum(heart_vertices, heart_faces, label_map,label_axis)

def load_procesed():

    DATASET = "valencia_real_pat1"
    m_vm , heart_vertices, heart_faces,fs =  load_data.loadDataSet(DATASET)
    load = np.load('resultados/'+DATASET+".npy", allow_pickle=True).item()
    label_x = np.array(load['label_x'])
    label_map = np.array(load['label_map'])
    representations.moving_spectrum(heart_vertices, heart_faces,label_map,label_x )

def load_MAFed():

    DATASET = "valencia_sintetic_la_fibrotic"
    m_vm , heart_vertices, heart_faces,fs =  load_data.loadDataSet(DATASET)
    t =  np.arange(0, m_vm.shape[1]/fs, 1/fs)
    f_axis,spectrum = representations.calculate_spectrum_matrix(t, m_vm)
    print(spectrum.shape)
    print(f_axis.shape)
    dif = np.abs(f_axis - 360)
    index = np.argmin(dif)
    spectrum = spectrum / np.max(np.abs(spectrum))
    spectrum = spectrum[:, 10:index]
    f_axis = f_axis[10:index]

    representations.moving_spectrum(heart_vertices, heart_faces,np.array(np.abs(spectrum)),np.array(f_axis))

    load = np.load('resultados/'+DATASET+".npy", allow_pickle=True).item()
    f_axis = np.array(load['f_axis'])
    FAM_spectrum = np.array(load['FAM_spectrum'])
    representations.moving_spectrum(heart_vertices, heart_faces,FAM_spectrum,f_axis)

def af_spatial_location(DATASET = "valencia_sintetic_la_fibrotic"):
    print("calculando:")
    #DATASET = "valencia_sintetic_la_fibrotic"
    m_vm , heart_vertices, heart_faces,fs =  load_data.loadDataSet(DATASET)
    #fs = 500

    t= np.arange(0,m_vm.shape[1])

    print(m_vm.shape)


    if m_vm.shape[1] %2 != 0:
        m_vm = m_vm[:,0:-1] 
        t = t[0:-1] 

    print(m_vm.shape)


    representations.plot_mesh(heart_vertices, heart_faces,m_vm,t , fs=fs , aux_signal=np.zeros(m_vm.shape))
    MAF_3D = maf.Maf()
    fo_map,peaks_map = MAF_3D.full_mesh_signal_proposal(m_vm, fs)
    fo_map = fo_map.reshape(-1, 1)

    print(fo_map.shape)
    
    representations.moving_spectrum(heart_vertices, heart_faces,fo_map,np.array([0]))
    representations.moving_spectrum(heart_vertices, heart_faces,peaks_map.reshape(-1,1),np.array([0]))

    print("aqui")
    solution = [1456,1481, 1505 ,1512 ,1516 ,1536 ,1539, 1541, 1563, 1566, 1568, 1576, 1583, 1603, 1613,
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

    map_solution = np.zeros(fo_map.shape)
    map_solution[solution] = 1
    print(solution)
    representations.moving_spectrum(heart_vertices, heart_faces,map_solution,np.array([0]))

    label_axis, label_map = spatial_analysis.full_spatial_proposal(heart_vertices,heart_faces,fo_map,6.4,5,5)

    representations.moving_spectrum(heart_vertices, heart_faces,label_map,np.array([0]))
    
    np.save("resultados/"+DATASET, {"fo_map":fo_map})



DATASET = "valencia_sintetic_ra_normal"
#initProgram()


af_spatial_location(DATASET)
#load_pipeline(DATASET,7)


#spectrograma()
#load_MAFed()
#load_procesed()




"""
Indices donde se ubica el rotor:


LA_NORMAL:

[1456,1481, 1505 ,1512 ,1516 ,1536 ,1539, 1541, 1563, 1566, 1568, 1576, 1583, 1603, 1613,
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


 
LA FIBROTIC:

[1340 ,1360  ,1439 ,1447 ,1456 ,1457 ,1487 ,
 1563 ,1566 ,1568 ,1603 ,1613 ,1617 ,1618 ,1626 ,1631 ,1646 ,1649,
 1652 ,1653 ,1661 ,1663 ,1664 ,1676 ,1680 ,1687 ,1688 ,1689 ,1700 ,1709 ,1717,
 1732 ,1741 ,1744 ,1746 ,1748 ,1751 ,1754 ,1761 ,1778 ,1780 ,1783 ,1795 ,1796 ,1797,
 1806 ,1808 ,1818 ,1823 ,1828 ,1829 ,1837 ,1841 ,1850 ,1851 ,1855 ,1859,
 1860 ,1861 ,1867 ,1868 ,1875 ,1877 ,1878 ,1880 ,1886 ,1891 ,1894 ,1898 ,1906 ,1907,
 1912 ,1916 ,1918 ,1919 ,1921 ,1923 ,1928 ,1930 ,1932 ,1939 ,1951 ,1959,
 1963 ,1964 ,1968 ,1976 ,1979 ,1981 ,1984 ,1986 ,1999 ,2001 ,2002,
 2006 ,2024]


RA NORMAL:

[270  ,274  ,281  ,301  ,303  ,308 ,322  ,327  ,332,
  334  ,337  ,345  ,346  ,351  ,353  ,359  ,364  ,368  ,378  ,380  ,387  ,390,
  391  ,400  ,405  ,420  ,421  ,422  ,424  ,431  ,433  ,435  ,436  ,438,
  445  ,447  ,450  ,467  ,480  ,482  ,485  ,488  ,494  ,496  ,497  ,499  ,501,
  505  ,517  ,521  ,523  ,539  ,542  ,550  ,551  ,557  ,565  ,567  ,568  ,571,
  578  ,582  ,583  ,586  ,591  ,598  ,601  ,605  ,611,
  612  ,613  ,616  ,617  ,625  ,629  ,630  ,632  ,634  ,636  ,638  ,640,
  652  ,653  ,662  ,668  ,669  ,675  ,678  ,697  ,699  ,708  ,718  ,723  ,724,
  729  ,764  ,767  ,772  ,773  ,789  ,794  ,810  ,822  ,842,
  848  ,860  ,866  ,868 ]

"""