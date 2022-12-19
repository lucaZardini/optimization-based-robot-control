

class Visual:
    '''
    Class representing one 3D mesh of the robot, to be attached to a joint. The class contains:
    * the name of the 3D objects inside Gepetto viewer.
    * the ID of the joint in the kinematic tree to which the body is attached.
    * the placement of the body with respect to the joint frame.
    This class is only used in the list Robot.visuals (see below).
    '''

    def __init__(self, name, jointParent, placement):
        self.name = name  # Name in gepetto viewer
        self.jointParent = jointParent  # ID (int) of the joint
        self.placement = placement  # placement of the body wrt joint, i.e. bodyMjoint

    def place(self, display, oMjoint):
        oMbody = oMjoint * self.placement
        display.place(self.name, oMbody, False)
