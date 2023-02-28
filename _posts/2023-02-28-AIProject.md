# LunarLander-v2
 
6조 - 김태준, 권건우, 송치윤


```python
import gym                       # OpenAI 에서 간단한 게임들을 통해서 강화학습을 테스트 할 수 있는 Gym 이라는 환경.
import collections               # episode를 stack하는 걸 도와주는 모듈을 사용. 후에 임의로 뽑음.
import random

import torch
import torch.nn as nn            # torch안에 nn이란 모듈을 사용(인공신경망 - neural network 형성을 도와줌)
import torch.nn.functional as F  # weight를 사용가능 .
import torch.optim as optim      # 여러 최적화 알고리즘을 수행하는 패키지. optimizer.step() 기능 수행.

#Hyperparameters
learning_rate = 0.0005            # 학습률, 얼마나 빨리할건지, 한걸음 얼마나 빠르게할건지
gamma         = 0.99              # discounting factor; 뒤쪽 reward는 덜
buffer_limit  = 50000             # 데이터 쌓을때, 5만개의 틀을 만들어 두고 5만개 이상의 데이터가 들어왔을때 가장 처음에 들어온것이 빠짐
batch_size    = 64                # DQN의 experience replay를 위한 batch.
env = gym.make('LunarLander-v2')  # OpenAI에서 구현되어 있는 여러 환경들 중 LunarLander-v2 게임 환경 load.
print('State shape: ', env.observation_space.shape)   # 환경의 feature의 갯수를 확인하기 위하여 print.
print('Number of actions: ', env.action_space.n)
```

    State shape:  (8,)
    Number of actions:  4
    


```python
class ReplayBuffer():             # collection에 stack하는 part.
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit) # collection안에 deque라는 method를 불러와 
                                                             # 50000개의 틀을 만들어 self.buffer에 넣어줌.
                                                             # deque는 stack과 queue의 기능을 모두 가진 객체. 리스트로 이해가능.
    def put(self, data):          # deque 안에 새로운 데이터를 넣어주는 역할
        self.buffer.append(data)
    
    def sample(self, n):          # epsiode로부터 sampling을 시작.
        mini_batch = random.sample(self.buffer, n)                            # random하게 뽑아 mini_batch 32개 생성.
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []  # 만들 mini_batch의 정보를 list안에 쪼개서 넣음.
                                                                              # done_mask는 boolean형태로 episode가 종료됬는지 알려줌.
        for transition in mini_batch:                                        # mini_batch에 matrix형태로 sampling한 data들을 넣는다.
            #미니 배치는 버퍼안의 몇개의 샘플을 뽑았다는 뜻.
            #샘플을 랜덤하게 뽑아서 미니 배치에 matrix 형태로 들어가 있다.
            #하나의 샘플은 벡터 형태로 저장되기 때문이다.
            # 64개의 샘플을 뽑아서 리스트 안에 하나씩 집어넣어준다.
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            #최종적으로 위 리스트 안에 배치 정보들이 쪼개져서 들어간다.

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)                    # 저장된 list 형태의 정보를 tensor 형태로 반환하여 torch계산이 가능하게함.
    
    def size(self):                                          # ReplayBuffer안에 size를 알려주는 것이 있어 size를 알려줌.
        return len(self.buffer)                              # 데이터 개수늘어날수록 버퍼 늘어나며 맥스 50000.
```


```python
class Qnet(nn.Module): # 인공신경망을 만들어준다; torch.nn.module = 모든 neural network module들의 기본적인 class.
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(8, 64)  # 8개의 node가 들어와 64개가 됨. state의 feature의 갯수는 8개이다. (ex.속도, 위치, 가속도 등)
        self.fc2 = nn.Linear(64, 64) # 입력과 출력 data를 제외하고 설정 가능함.
        self.fc3 = nn.Linear(64, 4)  # 64개에서 4개로 출력. action의 feature는 4개이다. (ex.위, 아래, 왼쪽, 오른쪽)

    def forward(self, x):            # node에 이어 activation function을 무엇으로 할지 결정. torch.nn.functional안의 relu를 사용하기로함
        x = F.relu(self.fc1(x))      # 1번째 layer를 relu로 activation function 사용.
        x = F.relu(self.fc2(x))      # 2번째 layer를 relu로 activation function 사용.
        x = self.fc3(x)              # 세번쨰 layer는 종착점, q값인데 음수가 나올 수 있어 그대로 사용
        return x
      
    def sample_action(self, obs, epsilon): # Epsilon - Greedy 법
        out = self.forward(obs)      # forward 함수에서 3번째 layer까지 거친 값이 out 으로 들어감.
        coin = random.random()       # 0과 1 임의로 뽑음
        if coin < epsilon:
            return random.randint(0,1) # 엡실론값보다 작은 coin값이 나올경우 q를 랜덤하게 뽑음 (모험)
        else : 
            return out.argmax().item() # out에는 4개의 q값이 나오는데, 가장 큰 q값을 greedy하게 뽑기
```


```python
def train(q, q_target, memory, optimizer):
    for i in range(40):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s) #s는 64*8 size이고 q는 64*4 (액션이 4개이므로)
        q_a = q_out.gather(1,a) #gather함수를 통하여 실제로 취했던 액션만 모은다.
        # gather 통과 후 q_a는 32*1 size이다.
        # epsilon-greedy법이기 때문에 완전히 최대의 q값을 모은것은 아니다.
        #q_a의 shape은 64*1이 된다.
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)  # 0과 1 2가지 action중에 max값을 취해라
        # target에서는 max값을 greedy하게 취한다.
        # unsqueeze는 차원을 1개 늘린다는 의미 (이후 곱셈연산을 위하여 차원을 맞춰주는 것)
        # s_prime은 64*8 shape이고 q_target으로 들어가면 64*4가 되고
        # unsqueeze를 통하여 64*1 된다.
        target = r + gamma * max_q_prime * done_mask
        # done mask는 끝나지 않은상태에선 1이지만 끝나면 0. (끝났을때는 target 업데이트 되면 안되므로)
        loss = F.smooth_l1_loss(q_a, target)                   # q_a와 target의 차이를 적용해서 loss func.을 쓰겟다.
        # -1~1 보다 큰 수는 제곱을하면 너무 커지므로 수렴문제가 있어서 적용하는 loss fn.
        optimizer.zero_grad()                                  #기울기 초기화
        loss.backward() 
        optimizer.step()                                       # back proposition하며 기울기 gradient를 계산(40번) -> w가 계속 update됨
```


```python
def main():
    q = Qnet()                               # behavior Q
    q_target = Qnet()                        # Qnet을 2번 부르는 이유 - 하나는 fixed 목적, 하나는 iteration 목적
    q_target.load_state_dict(q.state_dict()) # nn module안에 있음; q behavior가 가진 weight를 복사해서 q_targert에 삽입
                                             # 왜냐하면 q_target도 영원히 fix하는게 아니라, 업데이트가 필요하기 때문이다.
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    render = False                                   # 처음에 0이 들어가 있으며 후에 visualization을 위해 쓰임.
                                                      # 그림 그릴떄 쓰는 boolean. True가 되면 게임창이 나타난다.
    for n_epi in range(10000):                       # 강화학습에서는 초기에 모험을 많이 하게하려고 엡실론을 크게 한다.
        epsilon = max(0.01, 0.07 - 0.01*(n_epi/100)) # 에피소드가 거듭되며 엡실론이 작아진다.
        s = env.reset()                              # 초기에는 epsilon을 크게, 뒤에 가서는 작게 0.01로 고정.
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, done, info = env.step(a)     # done이 true면 done_mask를 실행
            done_mask = 0.0 if done else 1.0        # false면 done_mask에 1.0을 넣어 아직 게임이 종료되지 않음을 알려줌.
            memory.put((s,a,r,s_prime, done_mask))   # state의 변화를 메모리에 집어넣는다.
            s = s_prime

            score += r
            
            if render:                               # render가 false에서 true가 되면 활성화되어 우리에게 simulation을 보여줌
                env.render()
                
            if done:                                 # done이 true가 되지않는이상 계속 반복.
                break
            
        if score/print_interval >= 200:              # score가 원하는 결과값 이상일 때 render가 true가 됨.
            render = True
            
        if memory.size() > 2000:                     # 충분히 데이터가 쌓였을때 트레이닝 하기 위함이다.
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:    # (interval 초기값=20)으로 나눴을때 0이면, epi가 0이 아닐때 print.
            q_target.load_state_dict(q.state_dict()) # 20번을 거친 후에야 fix된 state를 update한다; q에서 복사.
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0                              # 20번마다 score를 reset한다.
    env.close()
```


```python
if __name__ == '__main__':
    main()
```

    n_episode :20, score : -139.0, n_buffer : 1456, eps : 6.8%
    n_episode :40, score : -194.6, n_buffer : 2866, eps : 6.6%
    n_episode :60, score : -151.7, n_buffer : 4213, eps : 6.4%
    n_episode :80, score : -143.2, n_buffer : 5623, eps : 6.2%
    n_episode :100, score : -134.9, n_buffer : 6996, eps : 6.0%
    n_episode :120, score : -137.6, n_buffer : 8360, eps : 5.8%
    n_episode :140, score : -137.1, n_buffer : 9653, eps : 5.6%
    n_episode :160, score : -144.3, n_buffer : 10975, eps : 5.4%
    n_episode :180, score : -136.6, n_buffer : 12339, eps : 5.2%
    n_episode :200, score : -123.4, n_buffer : 13743, eps : 5.0%
    n_episode :220, score : -119.6, n_buffer : 15104, eps : 4.8%
    n_episode :240, score : -112.0, n_buffer : 16578, eps : 4.6%
    n_episode :260, score : -102.1, n_buffer : 18073, eps : 4.4%
    n_episode :280, score : -90.4, n_buffer : 19967, eps : 4.2%
    n_episode :300, score : -101.3, n_buffer : 23379, eps : 4.0%
    n_episode :320, score : -188.5, n_buffer : 28115, eps : 3.8%
    n_episode :340, score : -177.8, n_buffer : 40193, eps : 3.6%
    n_episode :360, score : -185.6, n_buffer : 50000, eps : 3.4%
    n_episode :380, score : -211.3, n_buffer : 50000, eps : 3.2%
    n_episode :400, score : -216.4, n_buffer : 50000, eps : 3.0%
    n_episode :420, score : -137.9, n_buffer : 50000, eps : 2.8%
    n_episode :440, score : -191.1, n_buffer : 50000, eps : 2.6%
    n_episode :460, score : -224.4, n_buffer : 50000, eps : 2.4%
    n_episode :480, score : -189.9, n_buffer : 50000, eps : 2.2%
    n_episode :500, score : -204.9, n_buffer : 50000, eps : 2.0%
    n_episode :520, score : -175.6, n_buffer : 50000, eps : 1.8%
    n_episode :540, score : -155.5, n_buffer : 50000, eps : 1.6%
    n_episode :560, score : -135.0, n_buffer : 50000, eps : 1.4%
    n_episode :580, score : -151.2, n_buffer : 50000, eps : 1.2%
    n_episode :600, score : -130.6, n_buffer : 50000, eps : 1.0%
    n_episode :620, score : -151.0, n_buffer : 50000, eps : 1.0%
    n_episode :640, score : -110.4, n_buffer : 50000, eps : 1.0%
    n_episode :660, score : -123.4, n_buffer : 50000, eps : 1.0%
    n_episode :680, score : -111.7, n_buffer : 50000, eps : 1.0%
    n_episode :700, score : -99.1, n_buffer : 50000, eps : 1.0%
    n_episode :720, score : -93.3, n_buffer : 50000, eps : 1.0%
    n_episode :740, score : -85.5, n_buffer : 50000, eps : 1.0%
    n_episode :760, score : -82.1, n_buffer : 50000, eps : 1.0%
    n_episode :780, score : -75.5, n_buffer : 50000, eps : 1.0%
    n_episode :800, score : -78.1, n_buffer : 50000, eps : 1.0%
    n_episode :820, score : -70.2, n_buffer : 50000, eps : 1.0%
    n_episode :840, score : -70.3, n_buffer : 50000, eps : 1.0%
    n_episode :860, score : -59.9, n_buffer : 50000, eps : 1.0%
    n_episode :880, score : -48.5, n_buffer : 50000, eps : 1.0%
    n_episode :900, score : -27.9, n_buffer : 50000, eps : 1.0%
    n_episode :920, score : -46.8, n_buffer : 50000, eps : 1.0%
    n_episode :940, score : -47.9, n_buffer : 50000, eps : 1.0%
    n_episode :960, score : -52.0, n_buffer : 50000, eps : 1.0%
    n_episode :980, score : -48.1, n_buffer : 50000, eps : 1.0%
    n_episode :1000, score : -49.5, n_buffer : 50000, eps : 1.0%
    n_episode :1020, score : -61.4, n_buffer : 50000, eps : 1.0%
    n_episode :1040, score : -50.1, n_buffer : 50000, eps : 1.0%
    n_episode :1060, score : -51.4, n_buffer : 50000, eps : 1.0%
    n_episode :1080, score : -72.0, n_buffer : 50000, eps : 1.0%
    n_episode :1100, score : -37.4, n_buffer : 50000, eps : 1.0%
    n_episode :1120, score : -50.4, n_buffer : 50000, eps : 1.0%
    n_episode :1140, score : -58.9, n_buffer : 50000, eps : 1.0%
    n_episode :1160, score : -78.6, n_buffer : 50000, eps : 1.0%
    n_episode :1180, score : -71.7, n_buffer : 50000, eps : 1.0%
    n_episode :1200, score : -63.4, n_buffer : 50000, eps : 1.0%
    n_episode :1220, score : -51.6, n_buffer : 50000, eps : 1.0%
    n_episode :1240, score : -57.4, n_buffer : 50000, eps : 1.0%
    n_episode :1260, score : -42.2, n_buffer : 50000, eps : 1.0%
    n_episode :1280, score : -60.9, n_buffer : 50000, eps : 1.0%
    n_episode :1300, score : -56.5, n_buffer : 50000, eps : 1.0%
    n_episode :1320, score : -76.8, n_buffer : 50000, eps : 1.0%
    n_episode :1340, score : -65.0, n_buffer : 50000, eps : 1.0%
    n_episode :1360, score : -59.9, n_buffer : 50000, eps : 1.0%
    n_episode :1380, score : -54.9, n_buffer : 50000, eps : 1.0%
    n_episode :1400, score : -47.0, n_buffer : 50000, eps : 1.0%
    n_episode :1420, score : -30.3, n_buffer : 50000, eps : 1.0%
    n_episode :1440, score : -38.4, n_buffer : 50000, eps : 1.0%
    n_episode :1460, score : -35.4, n_buffer : 50000, eps : 1.0%
    n_episode :1480, score : -49.5, n_buffer : 50000, eps : 1.0%
    n_episode :1500, score : 5.9, n_buffer : 50000, eps : 1.0%
    n_episode :1520, score : -42.5, n_buffer : 50000, eps : 1.0%
    n_episode :1540, score : -21.7, n_buffer : 50000, eps : 1.0%
    n_episode :1560, score : -1.2, n_buffer : 50000, eps : 1.0%
    n_episode :1580, score : -37.2, n_buffer : 50000, eps : 1.0%
    n_episode :1600, score : -5.4, n_buffer : 50000, eps : 1.0%
    n_episode :1620, score : 22.8, n_buffer : 50000, eps : 1.0%
    n_episode :1640, score : -3.7, n_buffer : 50000, eps : 1.0%
    n_episode :1660, score : -9.2, n_buffer : 50000, eps : 1.0%
    n_episode :1680, score : 32.0, n_buffer : 50000, eps : 1.0%
    n_episode :1700, score : 6.5, n_buffer : 50000, eps : 1.0%
    n_episode :1720, score : -0.4, n_buffer : 50000, eps : 1.0%
    n_episode :1740, score : 17.9, n_buffer : 50000, eps : 1.0%
    n_episode :1760, score : -21.2, n_buffer : 50000, eps : 1.0%
    n_episode :1780, score : 22.4, n_buffer : 50000, eps : 1.0%
    n_episode :1800, score : 0.4, n_buffer : 50000, eps : 1.0%
    n_episode :1820, score : 17.9, n_buffer : 50000, eps : 1.0%
    n_episode :1840, score : 44.6, n_buffer : 50000, eps : 1.0%
    n_episode :1860, score : 61.5, n_buffer : 50000, eps : 1.0%
    n_episode :1880, score : 66.0, n_buffer : 50000, eps : 1.0%
    n_episode :1900, score : 51.8, n_buffer : 50000, eps : 1.0%
    n_episode :1920, score : 46.0, n_buffer : 50000, eps : 1.0%
    n_episode :1940, score : 54.6, n_buffer : 50000, eps : 1.0%
    n_episode :1960, score : 28.4, n_buffer : 50000, eps : 1.0%
    n_episode :1980, score : 45.8, n_buffer : 50000, eps : 1.0%
    n_episode :2000, score : 54.5, n_buffer : 50000, eps : 1.0%
    n_episode :2020, score : 53.7, n_buffer : 50000, eps : 1.0%
    n_episode :2040, score : 13.5, n_buffer : 50000, eps : 1.0%
    n_episode :2060, score : 96.7, n_buffer : 50000, eps : 1.0%
    n_episode :2080, score : 109.2, n_buffer : 50000, eps : 1.0%
    n_episode :2100, score : 115.3, n_buffer : 50000, eps : 1.0%
    n_episode :2120, score : 121.8, n_buffer : 50000, eps : 1.0%
    n_episode :2140, score : 90.5, n_buffer : 50000, eps : 1.0%
    n_episode :2160, score : 118.5, n_buffer : 50000, eps : 1.0%
    n_episode :2180, score : 149.1, n_buffer : 50000, eps : 1.0%
    n_episode :2200, score : 134.1, n_buffer : 50000, eps : 1.0%
    n_episode :2220, score : 143.8, n_buffer : 50000, eps : 1.0%
    n_episode :2240, score : 134.1, n_buffer : 50000, eps : 1.0%
    n_episode :2260, score : 135.7, n_buffer : 50000, eps : 1.0%
    n_episode :2280, score : 114.0, n_buffer : 50000, eps : 1.0%
    n_episode :2300, score : 121.1, n_buffer : 50000, eps : 1.0%
    n_episode :2320, score : 126.4, n_buffer : 50000, eps : 1.0%
    n_episode :2340, score : 152.8, n_buffer : 50000, eps : 1.0%
    n_episode :2360, score : 110.4, n_buffer : 50000, eps : 1.0%
    n_episode :2380, score : 165.5, n_buffer : 50000, eps : 1.0%
    n_episode :2400, score : 136.6, n_buffer : 50000, eps : 1.0%
    n_episode :2420, score : 135.4, n_buffer : 50000, eps : 1.0%
    n_episode :2440, score : 174.2, n_buffer : 50000, eps : 1.0%
    n_episode :2460, score : 172.2, n_buffer : 50000, eps : 1.0%
    n_episode :2480, score : 188.2, n_buffer : 50000, eps : 1.0%
    n_episode :2500, score : 205.7, n_buffer : 50000, eps : 1.0%
    n_episode :2520, score : 205.0, n_buffer : 50000, eps : 1.0%
    n_episode :2540, score : 210.7, n_buffer : 50000, eps : 1.0%
    


```python

```
