"""
Author: SpikaTech
Date: 01/09/2023
Description: This file contains all the functions related to all the spatial 3D analysis
(AF location and area segmentation) for the first proposal of AF detection and location.
This should be considered as the part 2 of the proposal as its expected to be connected to
the MAF model procedure.
"""

import numpy as np
from loguru import logger


def full_spatial_proposal(
    vertex,
    faces,
    fo_map,
    freq_threshold=7,
    dist_threshold=5,
    size_threshold=5,
):
    """
    Function that calculates all the process of analysing and segmentating (stages 3 and 4 of
    the algorithm).Do call this function to an MAF 3D output. This proposal does:

    1. Binarizes the fo map (fundamental frecuency map) using the freq_threshold param (usually 7hz).
       every point below that becomes 0.

    2.After that conected areas are segmented and labeled for further anlisys

    3.The resultant close areas are joined by a distance threshold (dist_threshold).
      After that, a size filter is applied to the areas to avoid "1 isolated point"
      error.

    4. Finally, all the resulting areas come together in just one binary color map which
       becomes the output of the algorithm. The most desired output its a completly
       empty map for non-AF cases and just one big segmented area for the AF cases.

    params:
        - vertex(np array): array with the coordenates of the vertex of the mesh in
                            a nvertex X 3 size.
        - faces (np array): array with the faces (triangles) of the surface of the mesh. Each face
                            contains 3 vertex indexes conected beetween them.

        - fo_map (np.array): array that contains a colormap made using the fo detector.

        - dist_threshold (int): This threshold is used to determine the distance beetween selected
                                bobs inthe segmentation process in order to select close, but not
                                conected bobs, as one.
        - size_threshold (int): Minimun number of conected points to be considered a blob to
                                segment.This is used to discard small not correlated points.
                                A high value is recomended.

    returns:

        - label_axis (np array): array with just one label ([0]) to represent the results

        - label_map (np array): spatial-label map that represents the selected interest areas.
                                its composed by just one vector of the size of the number of vertes

    """
    # Binarization by frecuency threshold
    fo_map[fo_map <= freq_threshold] = 0

    label_axis, label_map = segmentation(fo_map, vertex, faces, dist_threshold, size_threshold)

    if label_map.shape[1] != 0:
        label_map = np.where(label_map != 0, 1, 0)
        label_map = np.sum(label_map, axis=1).reshape(-1, 1)
        label_axis = np.array([0])

    return label_axis, label_map


def segmentation(color_map, vertex, faces, dist_threshold, size_threshold):
    """
    Function that does all the segmentation process, usually described as step 4 in the algorithm.

    params:
        - color_map (np array): vector that representates the color of the surface of the mesh
                                that is going to be segmented.
        - vertex (np array): array with the vertex of the mesh
        - faces (np array): array with the vertex conections (faces)
        - dist_threshold (int): maximun distance to consider two blobs as one.
        - size_threshold (int): maximun size ,size understanded as number of points contained in
                                one label, to be considered big enought to not be an error.

    returns:
        - label_axis (np array): vector that works as an label id axis for representation of results
        - label_map (np array): color map matrix where labels are extracted in each row, can be
                                represented with label_axis
    """
    label_axis, label_map, labels = connected_label_3d(color_map, vertex, faces)
    label_axis, label_map, labels = group_labels_by_distance(vertex, labels, dist_threshold)
    label_axis, label_map, labels = labels_size_filtering(vertex, labels, size_threshold)

    return label_axis, label_map


def connected_label_3d(color_map, vertex, faces):
    """
    Function that labels conected areas of a 3D color-map using a recreation of
    the 2D (images) algorithm "Connected-component labeling" from a colormap of a 3D surface.

    params:
        color_map (np array): vector with color values of a 3D mesh
        vertex (np array): Mesh vertex coordinates matrix
        faces (np array): Mesh faces (vertex conexions) matrix

    returns:
        label_axis (np array): vector that works as an label id axis for representation
                               of the results
        label_map (np array): color map matrix where labels are extracted in each row, can be
                              represented with label_axis
        labels (list): This function returns a list where each index represents a label. Each index
                       contains a sublist containing the vertex indices that belong to that label.
    """

    def label_point(point_index, faces, posible_index, label):
        """
        Sub-function that analyzes the neighbor points through their face connections and detects
        if they are marked.If a neighbor point is marked, the procedure is repeated for the new
        point, discarding the old one to ensure it will not be analyzed in the future.

        params:
            - point_index (int): actual index of the posible_index that is been analysed.
            - faces (np array): vertex-conections matrix.
            - posible_index (list): list with the index of the posible points (got marked , != 0)
                                    to analise if they have conections between them.
            - label (int): actual label index managing at the moment.

        returns:
            - labeled_array: list with actual point indexes for a label.
            - posible_index: updated list with the posible vertex to analyse.
            - new_faces (np array): updated face-array to avoid unnecesary repetitions
        """

        labeled_array = [point_index]
        new_faces = faces.copy()
        for i in np.arange(len(faces) - 1):
            if (point_index in posible_index) is False:
                break

            if point_index in faces[i]:
                for j in faces[i]:
                    if (j != point_index) and (j in posible_index):
                        subposible_index = posible_index.copy()
                        sub_faces = faces.copy()
                        sub_faces.remove(faces[i])
                        sub_labled_array, posible_index, new_faces = label_point(j, sub_faces, subposible_index, label)
                        labeled_array.append(sub_labled_array)

                try:
                    new_faces.remove(faces[i])
                except ValueError:
                    pass

        try:
            posible_index.remove(point_index)
        except ValueError:
            pass
        return labeled_array, posible_index, new_faces

    def flatten(lista):
        """
        Function that transforms an irregular dimention size list of list into one vector.
        params:
            - list (list): list.
        returns:
            - flattened (list): flattened list.
        """
        resultado = []
        for elemento in lista:
            if isinstance(elemento, list):
                resultado.extend(flatten(elemento))
            else:
                resultado.append(elemento)
        return resultado

    posible_array = (np.where(color_map != 0)[0]).tolist()
    faces = faces.tolist()
    segmentation_array = []
    label = 0
    while len(posible_array) != 0:
        point_index = posible_array[0]
        labeled_array, actualization, _ = label_point(point_index, faces, posible_array, label)
        segmentation_array.append(labeled_array)
        posible_array = actualization
        label += 1

    labels = []
    for i in segmentation_array:
        try:
            labels.append(list(flatten(i)))
        except ValueError:
            labels.append(i[0])

    label_axis, label_map = labels_2_colormap(vertex, labels)

    return label_axis, label_map, labels


def group_labels_by_distance(vertex, labels, dist_threshold):
    """
    Function that group diferent labels if they are close enought. This helps to prevent errors
    that generate disconection beetween areas.

    params:
        - vertex (np array): array with the vertex of the mesh
        - labels (list): array with the diferent vertex indexes of each label.
        - dist_threshold (int): maximun centroids distance to consider two blobs as one.

    returns:
        - label_axis (np array): vector that works as an label id axis for representation of the
                                 results
        - label_map (np array): color map matrix where labels are extracted in each row, can be
                                represented with label_axis
        - labels (list): This function returns a list where each index represents a label. Each
                        index contains a sublist containing the vertex indices that belong to that
                        label.
    """
    # Calculate centroid of each label
    labels_centroid = []
    for label in labels:
        mean = np.array([0.0, 0.0, 0.0])
        for i in label:
            mean += vertex[i, :]

        mean = (mean / len(label)).tolist()
        labels_centroid.append(mean)

    labels_centroid = np.array(labels_centroid)
    # calculate euclidean distance beetween centroids of labels
    centroid_map = np.zeros((len(labels_centroid), len(labels_centroid)))
    for i in np.arange(0, labels_centroid.shape[0]):
        for j in np.arange(0, labels_centroid.shape[0]):
            distance = np.linalg.norm(labels_centroid[i] - labels_centroid[j])
            if distance <= dist_threshold:
                centroid_map[i, j] = 1

    # Filter map and join labels
    centroid_map = np.unique(centroid_map, axis=0)
    centroid_map_merged = centroid_map.copy()
    max_sum_col = 2
    while max_sum_col > 1:
        for colum in centroid_map_merged.T:
            indexes = np.where(colum == 1)[0]
            if len(indexes) >= 2:
                for i in indexes[1:]:
                    centroid_map_merged[indexes[0]] += centroid_map_merged[i]

                centroid_map_merged = np.delete(centroid_map_merged, indexes[1:], axis=0)
                break

        centroid_map_merged = np.array(centroid_map_merged) > 0.9
        centroid_map_merged = centroid_map_merged.astype(int)
        max_sum_col = 1
        for colum in centroid_map_merged.T:
            sum_col = np.sum(colum)
            if max_sum_col < sum_col:
                max_sum_col = sum_col

    centroid_map = np.array(centroid_map_merged) > 0.9
    centroid_map = centroid_map.astype(int)
    new_labels = []
    for row in centroid_map:
        label_row = []
        for i in np.arange(row.shape[0]):
            if row[i] == 1:
                label_row += labels[i]
        new_labels.append(label_row)

    label_axis, label_map = labels_2_colormap(vertex, new_labels)

    return label_axis, label_map, new_labels


def labels_size_filtering(vertex, labels, size_threshold):
    """
    Function that filters small blobs as they are not considered as potetial locations of the
    FA. The size is considered to be the number of points that a label contains and not directly
    the area formed by the blob surface.

    params:
        - vertex (np array): array with the vertex of the mesh
        - labels (list): array with the diferent vertex indexes of each label.
        - size_threshold (int): minimun size of the blob (size being number of points contained by
                                 the label),to be considered as a valid blob. The input will the
                                 x thousand part of all points of the total surface

    returns:
        - label_axis (np array): vector that works as an label id axis for representation of the
                                results
        - label_map (np array): color map matrix where labels are extracted in each row, can be
                                represented with label_axis
        - labels (list): This function returns a list where each index represents a label. Each
                        index contains a sublist containing the vertex indices that belong to that
                        label.
    """

    size_threshold = int(vertex.shape[0] * size_threshold / 1000)
    filtered_labels = []
    for i in labels:
        if len(i) >= size_threshold:
            filtered_labels.append(i)

    if len(filtered_labels) == 0:
        logger.info("No AF detected")
        label_axis = np.array([0])
        label_map = np.zeros((vertex.shape[0], 1))
    else:
        label_axis, label_map = labels_2_colormap(vertex, filtered_labels)

    return label_axis, label_map, filtered_labels


def labels_2_colormap(vertex, labels):
    """
    Function that translates a label list into a full 3D color map representation of
    each label.

    params:
        - vertex (np array): array with the vertex of the mesh.
        - labels (list): array with the diferent vertex indexes of each label.

    returns:
        - label_axis (np array): vector that works as an label id axis for representation
                                 of the results
        - label_map (np array): color map matrix where labels are extracted in each row,
                                 can be represented with label_axis
    """

    label_axis = np.array([0])
    label_map = np.zeros((len(vertex), len(labels)))
    if len(labels) != 0:
        label_axis = np.arange(0, len(labels))
        for i in np.arange(0, label_map.shape[1]):
            to_add = np.zeros(len(vertex))
            to_add[labels[i]] = 1
            label_map[:, i] = to_add

    return label_axis, label_map
