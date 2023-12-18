import maf
import numpy as np
import load_data
import load_data
import matplotlib.pyplot as plt
import representations
import itertools
import spatial_analysis


#m_vm, heart_faces,heart_vertices, m_ve ,torso_faces,torso_vertices , torso_faces_cl,torso_vertices_cl = load_data.loadDataSet("utah_2018_sinus")


def initProgram():

 
    m_vm , heart_vertices, heart_faces =  load_data.loadDataSet("la_normal")
    #m_vm = np.delete(m_vm, -1, axis=1)
    fs = 498 #800 #actually is 500

    
    t =  np.arange(0, m_vm.shape[1]/fs, 1/fs) #np.linspace(start=0,stop=dur, num=int(dur * Fs) ) #t = 0:ts:dur-ts; % Segundos


    representations.spectrogram_3D(heart_vertices, heart_faces, m_vm, t,fs)

    FAM_3D = maf.Maf()
    FAM_spectrum, FAM_timeSignal,FAM_components, f_axis , t_axis = FAM_3D.maf_full_mesh(m_vm, fs , clean = True)

    representations.spectrogram_3D(heart_vertices, heart_faces, FAM_timeSignal, t_axis,fs)

    #print(FAM_components)
    #representations.showTimeMatrix(FAM_timeSignal,t_axis)
    maxFreqIndex =  int(len(f_axis)/12)
    MAF_fo = 0

    FAM_components = list(itertools.chain.from_iterable(FAM_components[0:]))
    components = np.unique(np.ravel(FAM_components).flatten())
    print(components)
    #representations.showSpectrumMatrix(FAM_spectrum[:, 0: maxFreqIndex],f_axis [0: maxFreqIndex])
    #representations.show_locatedFrecuency(f_axis,FAM_spectrum, 22.79 , heart_vertices, heart_faces)
    representations.moving_spectrum(heart_vertices, heart_faces, FAM_spectrum ,f_axis)
    np.save('atria_normal.npy', {"f_axis":f_axis, "FAM_spectrum" : FAM_spectrum, "FAM_timeSingal":FAM_timeSignal , "components":FAM_components ,"fo":MAF_fo })

def load_spectrum():


    m_vm , heart_vertices, heart_faces =  load_data.loadDataSet("la_normal")
    fs = 498 #800 #actually is 500
    t =  np.arange(0, m_vm.shape[1]/fs, 1/fs)
    f_axis0,spectrum = representations.calculate_spectrum_matrix(t,m_vm)
    print(f_axis0.shape)
    print(spectrum.shape)
    spectrum = np.abs(spectrum)[:,0:400]
    f_axis0 = f_axis0[0:400]
    spectrum = np.array(spectrum)
    print(f_axis0.shape)
    print(spectrum.shape)
    #representations.moving_spectrum(heart_vertices, heart_faces, spectrum,f_axis0)
    oct_signal, oct_f_axis = spatial_analysis.third_octaves_spectrum(spectrum,f_axis0)
    print(oct_signal.shape)
    print(oct_f_axis.shape)
    #representations.moving_spectrum(heart_vertices, heart_faces, oct_signal,oct_f_axis)


    #representations.textturized3D(heart_vertices)
    load = np.load('atria_normal_rare.npy', allow_pickle=True).item()
    f_axis1 = np.array(load['f_axis'])
    FAM_spectrum1 = np.array(load['FAM_spectrum'])
    FAM_spectrum1 = FAM_spectrum1 /(np.max(FAM_spectrum1))
    #FAM_spectrum1[FAM_spectrum1 > np.max(FAM_spectrum1)/4] = np.max(FAM_spectrum1)/4
    FAM1_comp = load['components']
    FAM1_fo = load['fo']

    load = np.load('atria_fibrotic.npy', allow_pickle=True).item()
    f_axis2 = np.array(load['f_axis'])
    FAM_spectrum2 = np.array(load['FAM_spectrum'])
    FAM_spectrum2 = FAM_spectrum2 /(np.max(FAM_spectrum2))
    #FAM_spectrum2[FAM_spectrum2 > np.max(FAM_spectrum2)/4] = np.max(FAM_spectrum2)/4
    FAM2_comp = load['components']
    FAM2_fo = load['fo']


    """RAW"""
    """
    m_vm , heart_vertices, heart_faces =  load_data_edgar.loadDataSet("la_normal")
    fs = 800
    t =  np.arange(0, m_vm.shape[1]/fs, 1/fs) #np.linspace(start=0,stop=dur, num=int(dur * Fs) ) #t = 0:ts:dur-ts; % Segundos
    f,spectrum = representations.calculateSpectrumMatrix(t,m_vm)
    sum_espectrum = np.sum(spectrum, axis=0)
    plt.plot(f[0:300],sum_espectrum[0:300])
    plt.show()
    
    

    sum_espectrum = np.sum(FAM_spectrum1, axis=0)
    plt.plot(f_axis1[0:300],sum_espectrum[0:300])
    plt.show()



    plt.hist(x=FAM1_comp, bins=np.arange(0,60), color='#F2AB6D', rwidth=0.5)
    plt.show()

    plt.hist(x=FAM2_comp, bins=np.arange(0,60), color='#F2AB6D', rwidth=0.5)
    plt.show()


    plt.hist(x=FAM1_fo, bins=np.arange(0,60), color='#F2AB6D', rwidth=0.5)
    plt.show()

    plt.hist(x=FAM2_fo, bins=np.arange(0,60), color='#F2AB6D', rwidth=0.5)
    plt.show()

    """

    maxFreqIndex =  int(len(f_axis2)/4)
    #freq interesantes: 19.2 22.79 ,26.39, 28.79* 31.19 , 33.59 , 45.59*


    mode = "normal"
    #over15 = MAF_representations.over15(heart_vertices, heart_faces, FAM_spectrum2[:, 0: maxFreqIndex] ,f_axis2[0: maxFreqIndex])
    #band_esp ,oct_f_axis = representations.oct_movingSpectrum(heart_vertices, heart_faces, FAM_spectrum2[:, 0: maxFreqIndex] ,f_axis1[0: maxFreqIndex],mode)
    #band_esp =  MAF_representations.movingSpectrum(heart_vertices, heart_faces, FAM_spectrum2[:, 0: maxFreqIndex] ,f_axis2[0: maxFreqIndex],mode)
    
    
    #representations.moving_spectrum(heart_vertices, heart_faces, FAM_spectrum1[:, 0: maxFreqIndex],f_axis2[0: maxFreqIndex])
    oct_signal, oct_f_axis = spatial_analysis.third_octaves_spectrum(FAM_spectrum2 ,f_axis2)
    #print(oct_signal.shape)
    print(oct_signal.shape)
    print(oct_f_axis.shape)
    representations.moving_spectrum(heart_vertices, heart_faces, oct_signal,oct_f_axis)
    threshold = np.max(np.abs(oct_signal))/5
    count_signal = spatial_analysis.repeated_freq(oct_f_axis,oct_signal,threshold,15)
    representations.moving_spectrum(heart_vertices, heart_faces, count_signal,[0])
    label_x, label_map, labels = spatial_analysis.connected_label_3d(count_signal,heart_vertices,heart_faces)
    label_x, label_map, labels = spatial_analysis.group_labels_by_distance(heart_vertices,labels,22)
    representations.moving_spectrum(heart_vertices, heart_faces, label_map,label_x)
    label_x, label_map, labels = spatial_analysis.labels_size_filtering(heart_vertices,labels,20)
    representations.moving_spectrum(heart_vertices, heart_faces, label_map,label_x)
    #label_x, label_map = spatial_analysis.segmentation_proposal(oct_f_axis,band_esp, heart_vertices,heart_faces ,threshold, 15,20,22)
    #MAF_representations.movingSpectrum(heart_vertices, heart_faces, label_x, label_map,"normal")
    
    #watershed_oct = MAF_representations.segmentation(oct_f_axis,band_esp,heart_vertices, heart_faces,threshold,15,22,20)


    5/0
    label_x, label_map = spatial_analysis.full_proposal(heart_vertices,
                                                                          heart_faces,
                                                                            FAM_spectrum1 ,
                                                                            f_axis1,
                                                                            0.25,15,60,22,20
                                                                            )
    print("calimoxo")
    print(label_map.shape)
    print(label_x.shape)
    representations.moving_spectrum(heart_vertices, heart_faces,label_map,label_x)

def af_spatial_location():
    print("calculando:")
    m_vm , heart_vertices, heart_faces =  load_data.loadDataSet("la_fibrotic")
    fs = 800
    MAF_3D = maf.Maf()
    FAM_spectrum, FAM_timeSignal,FAM_components, f_axis , t_axis = MAF_3D.maf_full_mesh(m_vm, fs , clean = True)
    label_x, label_map = spatial_analysis.full_proposal(heart_vertices,
                                                                          heart_faces,
                                                                            FAM_spectrum ,
                                                                            f_axis,
                                                                            0.25,15,60,22,20
                                                                            )
    print("Exit:")
    representations.moving_spectrum(heart_vertices, heart_faces,label_map,label_x )

initProgram()

#load_spectrum()

#af_spatial_location()

