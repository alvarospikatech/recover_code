import scipy.io as sci

general_path = "src/VRCARDIO-Testing/results/controlled/"


def loadDataSet(value):

    if value == "utah_2002_03":
        # Potencial corazon
        m_vm = sci.loadmat(general_path + "utah_2003/rsm15may02-cage-0003.mat")["ts"]["potvals"][
            0, 0
        ]  # Transmembrana ("""corazon""")
        # Mesh corazon
        readcagemesh = sci.loadmat(general_path + "utah_2003/cage.mat")
        heart_vertices = readcagemesh["cage"]["node"][0, 0]
        heart_faces = readcagemesh["cage"]["face"][0, 0]  # julia/matlab - python offset
        # Potencial Medido Torso
        m_ve = sci.loadmat(general_path + "/utah_2003/rsm15may02-ts-0003.mat")["ts"]["potvals"][
            0, 0
        ]  # Extracelular (Nodos)
        # mesh cage(torso)
        readtorsomesh = sci.loadmat(general_path + "utah_2003/tank_192.mat")
        torso_vertices = readtorsomesh["tank"]["pts"][0, 0]  # Vertices de puntos malla torso [x,y,z] --> (3,192)
        torso_faces = readtorsomesh["tank"]["fac"][0, 0]  # Conectividad malla torso (triangulos) [P1,P2,P3] --> (3,352)

        # closed mesh
        readtorsomesh = sci.loadmat(general_path + "utah_2003/tank_771_closed.mat")
        torso_vertices_cl = readtorsomesh["torso"]["node"][0, 0]  # Vertices de puntos malla torso [x,y,z] --> (3,192)
        torso_faces_cl = readtorsomesh["torso"]["face"][
            0, 0
        ]  # Conectividad malla torso (triangulos) [P1,P2,P3] --> (3,352)

        torso_faces = torso_faces - 1
        torso_faces_cl = torso_faces_cl - 1
        heart_faces = heart_faces - 1

        torso_vertices = torso_vertices.T
        torso_faces = torso_faces.T

        torso_vertices_cl = torso_vertices_cl.T
        torso_faces_cl = torso_faces_cl.T

        heart_vertices = heart_vertices.T
        heart_faces = heart_faces.T

    elif value == "utah_2002_33":

        # Potencial corazon
        m_vm = sci.loadmat(general_path + "utah/rsm15may02-cage-0033.mat")["ts"]["potvals"][
            0, 0
        ]  # Transmembrana ("""corazon""")
        # Mesh corazon
        readcagemesh = sci.loadmat("utah/cage.mat")
        heart_vertices = readcagemesh["cage"]["node"][0, 0]
        heart_faces = readcagemesh["cage"]["face"][0, 0]  # julia/matlab - python offset
        # Potencial Medido Torso
        m_ve = sci.loadmat(general_path + "Data/utah/rsm15may02-ts-0033.mat")["ts"]["potvals"][
            0, 0
        ]  # Extracelular (Nodos)
        # mesh cage(torso)
        readtorsomesh = sci.loadmat(general_path + "utah/tank_192.mat")
        torso_vertices = readtorsomesh["tank"]["pts"][0, 0]  # Vertices de puntos malla torso [x,y,z] --> (3,192)
        torso_faces = readtorsomesh["tank"]["fac"][0, 0]  # Conectividad malla torso (triangulos) [P1,P2,P3] --> (3,352)

        # closed mesh
        readtorsomesh = sci.loadmat(general_path + "utah/tank_771_closed.mat")
        torso_vertices_cl = readtorsomesh["torso"]["node"][0, 0]  # Vertices de puntos malla torso [x,y,z] --> (3,192)
        torso_faces_cl = readtorsomesh["torso"]["face"][
            0, 0
        ]  # Conectividad malla torso (triangulos) [P1,P2,P3] --> (3,352)

        torso_faces = torso_faces - 1
        torso_faces_cl = torso_faces_cl - 1
        heart_faces = heart_faces - 1

        torso_vertices = torso_vertices.T
        torso_faces = torso_faces.T

        torso_vertices_cl = torso_vertices_cl.T
        torso_faces_cl = torso_faces_cl.T

        heart_vertices = heart_vertices.T
        heart_faces = heart_faces.T

    elif value == "utah_2018_avp":

        # Potencial corazon
        m_vm = sci.loadmat(general_path + "utah_2018/cageBeat_avp.mat")["ts"]["potvals"][
            0, 0
        ]  # Transmembrana ("""corazon""")
        # Mesh corazon
        readcagemesh = sci.loadmat(general_path + "utah_2018/cageGeom.mat")

        heart_vertices = readcagemesh["cageGeom"]["node"][0, 0]
        heart_faces = readcagemesh["cageGeom"]["face"][0, 0]  # julia/matlab - python offset
        # Potencial Medido Torso
        m_ve = sci.loadmat(general_path + "utah_2018/torsoBeat_avp.mat")["ts"]["potvals"][0, 0]  # Extracelular (Nodos)
        # mesh cage(torso)
        readtorsomesh = sci.loadmat(general_path + "utah_2018/torsoGeom_measurements.mat")
        torso_vertices = readtorsomesh["torsoGeom_measurements"]["node"][
            0, 0
        ]  # Vertices de puntos malla torso [x,y,z] --> (3,192)
        torso_faces = readtorsomesh["torsoGeom_measurements"]["face"][
            0, 0
        ]  # Conectividad malla torso (triangulos) [P1,P2,P3] --> (3,352)

        # closed mesh
        readtorsomesh = sci.loadmat(general_path + "utah_2018/torsoGeom_closed.mat")
        torso_vertices_cl = readtorsomesh["torsoGeom_closed"]["node"][
            0, 0
        ]  # Vertices de puntos malla torso [x,y,z] --> (3,192)
        torso_faces_cl = readtorsomesh["torsoGeom_closed"]["face"][
            0, 0
        ]  # Conectividad malla torso (triangulos) [P1,P2,P3] --> (3,352)

        torso_faces = torso_faces - 1
        torso_faces_cl = torso_faces_cl - 1
        heart_faces = heart_faces - 1

        torso_vertices = torso_vertices.T
        torso_faces = torso_faces.T

        torso_vertices_cl = torso_vertices_cl.T
        torso_faces_cl = torso_faces_cl.T

        heart_vertices = heart_vertices.T
        heart_faces = heart_faces.T

    elif value == "utah_2018_pvp":

        # Potencial corazon
        m_vm = sci.loadmat(general_path + "utah_2018/cageBeat_pvp.mat")["ts"]["potvals"][
            0, 0
        ]  # Transmembrana ("""corazon""")
        # Mesh corazon
        readcagemesh = sci.loadmat(general_path + "utah_2018/cageGeom.mat")

        heart_vertices = readcagemesh["cageGeom"]["node"][0, 0]
        heart_faces = readcagemesh["cageGeom"]["face"][0, 0]  # julia/matlab - python offset
        # Potencial Medido Torso
        m_ve = sci.loadmat(general_path + "utah_2018/torsoBeat_pvp.mat")["ts"]["potvals"][0, 0]  # Extracelular (Nodos)
        # mesh cage(torso)
        readtorsomesh = sci.loadmat(general_path + "utah_2018/torsoGeom_measurements.mat")
        torso_vertices = readtorsomesh["torsoGeom_measurements"]["node"][
            0, 0
        ]  # Vertices de puntos malla torso [x,y,z] --> (3,192)
        torso_faces = readtorsomesh["torsoGeom_measurements"]["face"][
            0, 0
        ]  # Conectividad malla torso (triangulos) [P1,P2,P3] --> (3,352)

        # closed mesh
        readtorsomesh = sci.loadmat(general_path + "Meshes/utah_2018/torsoGeom_closed.mat")
        torso_vertices_cl = readtorsomesh["torsoGeom_closed"]["node"][
            0, 0
        ]  # Vertices de puntos malla torso [x,y,z] --> (3,192)
        torso_faces_cl = readtorsomesh["torsoGeom_closed"]["face"][
            0, 0
        ]  # Conectividad malla torso (triangulos) [P1,P2,P3] --> (3,352)

        torso_faces = torso_faces - 1
        torso_faces_cl = torso_faces_cl - 1
        heart_faces = heart_faces - 1

        torso_vertices = torso_vertices.T
        torso_faces = torso_faces.T

        torso_vertices_cl = torso_vertices_cl.T
        torso_faces_cl = torso_faces_cl.T

        heart_vertices = heart_vertices.T
        heart_faces = heart_faces.T

    elif value == "utah_2018_sinus":

        # Potencial corazon
        m_vm = sci.loadmat(general_path + "utah_2018/cageBeat_sinus.mat")["ts"]["potvals"][
            0, 0
        ]  # Transmembrana ("""corazon""")
        # Mesh corazon
        readcagemesh = sci.loadmat(general_path + "utah_2018/cageGeom.mat")

        heart_vertices = readcagemesh["cageGeom"]["node"][0, 0]
        heart_faces = readcagemesh["cageGeom"]["face"][0, 0]  # julia/matlab - python offset
        # Potencial Medido Torso
        m_ve = sci.loadmat(general_path + "Data/utah_2018/torsoBeat_sinus.mat")["ts"]["potvals"][
            0, 0
        ]  # Extracelular (Nodos)
        # mesh cage(torso)
        readtorsomesh = sci.loadmat(general_path + "utah_2018/torsoGeom_measurements.mat")
        torso_vertices = readtorsomesh["torsoGeom_measurements"]["node"][
            0, 0
        ]  # Vertices de puntos malla torso [x,y,z] --> (3,192)
        torso_faces = readtorsomesh["torsoGeom_measurements"]["face"][
            0, 0
        ]  # Conectividad malla torso (triangulos) [P1,P2,P3] --> (3,352)

        # closed mesh
        readtorsomesh = sci.loadmat(general_path + "utah_2018/torsoGeom_closed.mat")
        torso_vertices_cl = readtorsomesh["torsoGeom_closed"]["node"][
            0, 0
        ]  # Vertices de puntos malla torso [x,y,z] --> (3,192)
        torso_faces_cl = readtorsomesh["torsoGeom_closed"]["face"][
            0, 0
        ]  # Conectividad malla torso (triangulos) [P1,P2,P3] --> (3,352)

        torso_faces = torso_faces - 1
        torso_faces_cl = torso_faces_cl - 1
        heart_faces = heart_faces - 1

        torso_vertices = torso_vertices.T
        torso_faces = torso_faces.T

        torso_vertices_cl = torso_vertices_cl.T
        torso_faces_cl = torso_faces_cl.T

        heart_vertices = heart_vertices.T
        heart_faces = heart_faces.T

    elif value == "maastrich_sinus":

        # Potencial corazon
        m_vm = sci.loadmat(general_path + "maastrich/heartpots_sinus.mat")["hartpots"]  # Transmembrana ("""corazon""")
        # Mesh corazon
        readcagemesh = sci.loadmat(general_path + "maastrich/heart_sinus.mat")["hart"]
        heart_vertices = readcagemesh[0][0][0]
        heart_faces = readcagemesh[0][0][1]
        heart_faces = heart_faces - 1  # julia/matlab - python offset
        # Potencial Medido Torso
        m_ve = sci.loadmat(general_path + "maastrich/bodypots_sinus.mat")["lichaampots"]  # Extracelular (Nodos)
        # mesh cage(torso)
        readtorsomesh = sci.loadmat(general_path + "maastrich/body_sinus.mat")["lichaam"]
        torso_vertices = readtorsomesh[0][0][0]  # Vertices de puntos malla torso [x,y,z] --> (3,192)
        torso_faces = readtorsomesh[0][0][1]  # Conectividad malla torso (triangulos) [P1,P2,P3] --> (3,352)
        torso_faces = torso_faces - 1  # julia/matlab - python offset

        torso_vertices_cl = torso_vertices
        torso_faces_cl = torso_faces

        m_ve = m_ve / 1000  # Ajuste dimensiones de mV a V
        m_vm = m_vm / 1000

    elif value == "maastrich_paced":

        # Potencial corazon
        m_vm = sci.loadmat(general_path + "maastrich/heartpots_paced.mat")["hartpots"]  # Transmembrana ("""corazon""")
        # Mesh corazon
        readcagemesh = sci.loadmat(general_path + "maastrich/heart_paced.mat")["hart"]
        heart_vertices = readcagemesh[0][0][0]
        heart_faces = readcagemesh[0][0][1]
        heart_faces = heart_faces - 1  # julia/matlab - python offset
        # Potencial Medido Torso
        m_ve = sci.loadmat(general_path + "maastrich/bodypots_paced.mat")["lichaampots"]  # Extracelular (Nodos)
        # mesh cage(torso)
        readtorsomesh = sci.loadmat(general_path + "maastrich/body_paced.mat")["lichaam"]
        torso_vertices = readtorsomesh[0][0][0]  # Vertices de puntos malla torso [x,y,z] --> (3,192)
        torso_faces = readtorsomesh[0][0][1]  # Conectividad malla torso (triangulos) [P1,P2,P3] --> (3,352)
        torso_faces = torso_faces - 1  # julia/matlab - python offset
        m_ve = m_ve / 1000  # Ajuste dimensiones de mV a V
        m_vm = m_vm / 1000
        torso_vertices_cl = torso_vertices
        torso_faces_cl = torso_faces

    else:
        print("No proper dataset selected")

    return m_vm, heart_faces, heart_vertices, m_ve, torso_faces, torso_vertices, torso_faces_cl, torso_vertices_cl
