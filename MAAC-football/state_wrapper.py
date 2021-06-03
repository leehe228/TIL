import numpy as np

def make_state(obs):

    state = []

    for p in range(3):
        o = obs[p]
        tempList = []
        # activate and designated
        if o['active'] == p:
            tempList.append(1)
        else:
            tempList.append(0)

        if o['designated'] == p:
            tempList.append(1)
        else:
            tempList.append(0)    

        # home team
        tempList.append(o['left_team_tired_factor'][p])
        tempList.append(o['left_team'][p][0])
        tempList.append(o['left_team'][p][1])

        for l in range(len(o['left_team'])):
            if l == p: continue
            else:
                tempList.append(o['left_team'][l][0])
                tempList.append(o['left_team'][l][1])

        tempList.append(o['left_team_direction'][p][0])
        tempList.append(o['left_team_direction'][p][1])

        for l in range(len(o['left_team_direction'])):
            if l == p: continue
            else:
                tempList.append(o['left_team_direction'][l][0])
                tempList.append(o['left_team_direction'][l][1])
        
        # side team
        for r in o['right_team']:
            tempList.append(r[0])
            tempList.append(r[1])
        for r in o['right_team_direction']:
            tempList.append(r[0])
            tempList.append(r[1])

        # ball
        tempList.append(o['ball'][0])
        tempList.append(o['ball'][1])
        tempList.append(o['ball'][2])
        tempList.append(o['ball_rotation'][0])
        tempList.append(o['ball_rotation'][1])
        tempList.append(o['ball_rotation'][2])
        tempList.append(o['ball_direction'][0])
        tempList.append(o['ball_direction'][1])
        tempList.append(o['ball_direction'][2])
        tempList.append(o['ball_owned_team'])

        # etc
        tempList.append(o['score'][0])
        tempList.append(o['score'][1])
        tempList.append(o['game_mode'])

        state.append(np.array(tempList))

    state = np.array(state)
    
    return state