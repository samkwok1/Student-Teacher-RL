import sys
import numpy as np
import pandas
import random
import time
class Q_learning():
    def __init__(self, num_states, num_actions, gamma, alpha):
        self.Q = np.zeros((num_states, num_actions))
        self.gamma = gamma
        self.alpha = alpha
    def init_rewards(infile):
        r_given_state_action = {}
        indicate = 0
        s_ret = 0
        lines = []
        line_num = 0
        with open(infile) as f:
            f.readline()[1:]
            for line in f:
            s, a, r, sp = line.split(',')
            s, a, r, sp = int(s) - 1, int(a) - 1, int(r), int(sp) - 1
            lines.append(s)
            if s not in r_given_state_action:
                r_given_state_action[s] = []
            r_given_state_action[s].append((line_num, r, a, sp))
            if indicate == 0:
                indicate += 1
                s_ret = s
            line_num += 1
        for key, value in r_given_state_action.items():
            value = sorted(value, key = lambda x: x[0])
        return r_given_state_action, s_ret, lines
    def update_rewards(infile, num_states, num_actions, gamma, alpha, epsilon):
        r_given_next_state, s, lines = init_rewards(infile)
        model = Q_learning(num_states, num_actions, gamma, alpha)
        r = 0
        s_orig = s
        sp = s_orig
        episode = 0
        num_lines = model.Q.shape[0]
        while episode < 10:
            s = s_orig
            line = 0
            for i in range(num_lines * 2):  
                if sp not in r_given_next_state.keys():
                    sp = lines[line]
                s = sp
                possible_actions_rewards = r_given_next_state[s]
                for i in range(len(possible_actions_rewards)):
                    if possible_actions_rewards[i][0] == line:
                        r = possible_actions_rewards[i][1]
                        a = possible_actions_rewards[i][2]
                        sp = possible_actions_rewards[i][3]
                        break
                gamma, Q, alpha = model.gamma, model.Q, model.alpha
                next_state_loc = model.Q[sp]
                max_r = next_state_loc.max()
                if_not_taken = Q[s, a]
                model.Q[s, a] += alpha * (r + gamma * max_r - if_not_taken)
                line += 1
            episode += 1
        return model
    def create_policy(model, policies, index):
        policy = (model.Q[index]).argmax()
        policies.append(policy + 1)
    def main():
        infile, outfile, states, actions = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        gamma, alpha, epsilon = float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
        start = time.time()
        model = update_rewards(infile, states, actions, gamma, alpha, epsilon)
        final_policy = []
        for i in range(states):
            create_policy(model, final_policy, i)
        df = pandas.DataFrame(final_policy)
        df.to_csv(outfile, index=False)
        end = time.time()
        print(end - start)

    if __name__ == '__main__':
        main()