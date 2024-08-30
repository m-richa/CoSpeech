import mediapipe.python.solutions.face_mesh_connections as FaceMesh  # google mediapipe

FACEMESH_NOSE_CONNECTION = frozenset([(188,174),(174,236),(236,198),(198,49),(49,102),(102,64),
                           (64,240),(240,97),(97,2),(2,326),(326,460),(460,294),(294,331),
                           (331,279),(279,420),(420,456),(456,399),(399,412)])

# FACEMESH_NOSE = frozenset([(188,174),(174,236),(236,198),(198,49)])

FACEMESH_CONDITIONAL_FEATURE_MAP_CONNECTIONS =  frozenset().union(*[FaceMesh.FACEMESH_CONTOURS, FACEMESH_NOSE_CONNECTION])

BODY_POSE_CONNECTIONS = frozenset([(11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])

BODY_POSE_NO_HAND_CONNECTIONS = frozenset([(11, 12), (11, 13),
                              (13, 15),
                              (12, 14), (14, 16),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])
FACEMESH_NOSE = frozenset([(188,174),(174,236),(236,198),(198,49),(49,102),(102,64),(64,240),(240,97),(97,2),(2,326),(326,460),(460,294),(294,331),(331,279),(279,420),(420,456),(456,399),(399,412)])

FACEMESH_CONDITIONAL_FEATURE_MAP_CONNECTIONS =  frozenset().union(*[FaceMesh.FACEMESH_CONTOURS, FACEMESH_NOSE])