

# create artificially hip, spine, thorax, headbase, headtop
def artficial_kps(nose, lhip, rhip, rsh, lsh):
    hipx = (lhip[0] + rhip[0]) / 2
    hipy = (lhip[1] + rhip[1]) / 2
    hip = [hipx, hipy]

    thoraxx = (rsh[0] + lsh[0]) / 2
    thoraxy = (rsh[1] + lsh[1]) / 2
    thorax = [thoraxx, thoraxy]

    headbasex = (thorax[0] + nose[0]) / 2
    headbasey = (thorax[1] + nose[1]) / 2
    headbase = [headbasex, headbasey]

    headtopx = -thorax[0] + 2*nose[0]
    headtopy = -thorax[1] + 2*nose[1]
    headtop = [headtopx, headtopy]

    spinex = (headbase[0] + hip[0]) / 2
    spiney = (headbase[1] + hip[1]) / 2
    spine = [spinex, spiney]
    return hip, spine, thorax, headbase, headtop


def artficial_joints3d(nose, lhip, rhip, rsh, lsh):
    hipx = (lhip[0] + rhip[0]) / 2
    hipy = (lhip[1] + rhip[1]) / 2
    hipz = (lhip[2] + rhip[2]) / 2
    hip = [hipx, hipy, hipz]

    thoraxx = (rsh[0] + lsh[0]) / 2
    thoraxy = (rsh[1] + lsh[1]) / 2
    thoraxz = (rsh[2] + lsh[2]) / 2
    thorax = [thoraxx, thoraxy, thoraxz]

    headbasex = (thorax[0] + nose[0]) / 2
    headbasey = (thorax[1] + nose[1]) / 2
    headbasez = (thorax[2] + nose[2]) / 2
    headbase = [headbasex, headbasey,headbasez]

    headtopx = -thorax[0] + 2*nose[0]
    headtopy = -thorax[1] + 2*nose[1]
    headtopz = headbase[2]
    headtop = [headtopx, headtopy, headtopz]

    spinex = (headbase[0] + hip[0]) / 2
    spiney = (headbase[1] + hip[1]) / 2
    spinez = thorax[2]
    spine = [spinex, spiney, spinez]

    return hip, spine, thorax, headbase, headtop



