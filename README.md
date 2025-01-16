This is the repository for final project for Psych240A: Curiosity in AI @ Stanford.
---

## **Abstract**
Reinforcement learning has been shown to align with many aspects of human learning. However, the impact of different teaching strategies on a learner's performance is not yet fully understood. In this project, we explore how the timing of advice (pre-advice vs. post-advice) and the reliability of the teacher affect the learning efficiency of students in a maze navigation task. We train a parent agent to learn an optimal policy, then systematically vary its reliability from 0\% to 100\% by scrambling the learned Q-table. Child agents are then trained with the parent's guidance using either pre-advice (parent recommends action before child decides) or post-advice (parent advice is incorporated after child takes action) strategies. Learning efficiency is measured by the number of episodes needed for the child's Q-table to converge to the optimal policy. Results show that in the pre-advice condition, the child's learning is largely unaffected by the parent's reliability. In the post-advice condition, however, lower parent reliability significantly hinders the child's ability to converge to the optimal policy. These findings suggest that the timing of advice and the teacher's reliability have substantial impacts on the learner's outcomes in reinforcement learning settings. Understanding these effects paves the way for developing more effective teaching strategies for both human and AI learners.

paper: [Student-Teacher Reinforcement Learning](https://drive.google.com/drive/u/1/folders/1Xxbrqjswo8zsHHTPSZPXfLwTMSDUOK_z)

---

## **Repository Structure**
```plaintext
src/
│
├── algo/                 
│   └── q.py              # Contains the q-learning algorithm
│
├── config/               # Contains the config file where hyperparameters are set 
│
├── environment/
│   ├── maze.py           # Generates the maze
│   └── reward_maze.py    # Generates the reward maze based on the maze structure
│
├── util/                 
│   └── plots.py      
│                          
└── main.py                # All functionality runs through main

```
email: [samkwok@stanford.edu](mailto:samkwok@stanford.edu)

Reach out with questions!
