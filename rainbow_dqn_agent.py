import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from city_env import CityEnv
import time

# --- 1. ARCHITEKTURA SIECI NEURONOWEJ ---
class DuelingQNetwork(nn.Module):
    def __init__(self, grid_size=50, metrics_size=11, action_size=15001):
        super(DuelingQNetwork, self).__init__()
        
        input_dim = (grid_size * grid_size) + metrics_size
        
        # 1. Wspólny Ekstraktor Cech (Feature Extractor)
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU()
        )
        
        # 2. Strumień Wartości V(s) - Wypuszcza 1 wartość (ocena stanu)
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 3. Strumień Przewagi A(s, a) - Wypuszcza wartości dla każdej akcji (15001)
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        
    def forward(self, grid, metrics):
        grid_flat = grid.view(grid.size(0), -1)
        x = torch.cat((grid_flat, metrics), dim=1)
        
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Specjalna agregacja Dueling DQN: Q(s,a) = V(s) + (A(s,a) - średnia z A)
        # Odejmujemy średnią dla stabilności matematycznej gradientów
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
# --- 2A. STRUKTURA SUMTREE (Błyskawiczne szukanie priorytetów) ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # Drzewo ma 2 * capacity - 1 węzłów (liście na samym dole przechowują priorytety)
        self.tree = np.zeros(2 * capacity - 1)
        # Tablica na fizyczne dane (stany, akcje, nagrody)
        self.data = np.zeros(capacity, dtype=object)
        self.write_ptr = 0
        self.n_entries = 0

    def add(self, priority, data):
        # Indeks liścia w drzewie
        tree_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)
        
        self.write_ptr += 1
        if self.write_ptr >= self.capacity:
            self.write_ptr = 0 # Nadpisywanie starych danych
            
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        # Różnica między nowym a starym priorytetem
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagacja zmiany w górę drzewa
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
                
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0] # Korzeń drzewa trzyma sumę wszystkich priorytetów

# --- 2B. PRIORITIZED EXPERIENCE REPLAY (PER) ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity=20000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.alpha = alpha # Jak mocno ufamy priorytetom (0 = losowe, 1 = tylko najwyższe)
        self.beta = beta   # Korekta wagi dla propagacji wstecznej
        self.beta_increment = beta_increment
        self.epsilon = 1e-5 # Mała stała, by żaden priorytet nie wynosił równe 0

    def push(self, state, action, reward, next_state, done):
        # Nowe doświadczenia zawsze wpadają z MAKSYMALNYM priorytetem, 
        # by sieć na pewno na nie spojrzała chociaż raz
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
            
        experience = (state, action, reward, next_state, done)
        self.tree.add(max_priority, experience)

    def sample(self, batch_size):
        batch = []
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Obliczamy wagi IS (Importance Sampling)
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        
        # Dzielimy przedział priorytetów na segmenty dla równomiernego losowania
        segment = self.tree.total_priority / batch_size
        
        # Prawdopodobieństwo minimalne w drzewie do skalowania wag
        p_min = np.min(self.tree.tree[-self.tree.capacity: -self.tree.capacity + self.tree.n_entries]) / self.tree.total_priority
        if p_min == 0: p_min = 1e-5
        max_weight = (p_min * self.tree.n_entries) ** (-self.beta)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            
            tree_idx, priority, data = self.tree.get_leaf(v)
            
            priorities[i] = priority
            indices[i] = tree_idx
            batch.append(data)
            
            # Wzór na wagę Importance Sampling
            sampling_prob = priority / self.tree.total_priority
            weights[i] = np.power(self.tree.n_entries * sampling_prob, -self.beta) / max_weight
            
        return batch, indices, weights

    def update_priorities(self, tree_indices, td_errors):
        priorities = np.power(np.abs(td_errors) + self.epsilon, self.alpha)
        for idx, priority in zip(tree_indices, priorities):
            self.tree.update(idx, priority)
            
    def __len__(self):
        return self.tree.n_entries

# --- 3. KLASA AGENTA ---
class DQNAgent:
    def __init__(self, grid_size=50, metrics_size=11, action_size=15001):
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Używamy nowej architektury Dueling
        self.policy_net = DuelingQNetwork(grid_size, metrics_size, action_size).to(self.device)
        self.target_net = DuelingQNetwork(grid_size, metrics_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005) # Delikatnie niższy Learning Rate
        self.memory = PrioritizedReplayBuffer(capacity=20000)
        
        self.batch_size = 128
        self.gamma = 0.99       
        self.epsilon = 1.0      
        self.epsilon_min = 0.1  
        self.epsilon_decay = 0.97

    def select_action(self, state, env): # TUTAJ znajduje się dodany parametr env
        # 1. Eksploracja losowa (ale mądra - losujemy do skutku aż trafimy na puste pole lub No-Op)
        if np.random.rand() <= self.epsilon:
            while True:
                action = random.randrange(self.action_size)
                if action == self.action_size - 1: 
                    return action # No-Op (czekanie) jest zawsze legalne
                
                x, z, _, _ = env._decode_action(action)
                if state['map_grid'][x][z] == 0: 
                    return action
                    
        # 2. Eksploatacja (Wiedza z sieci)
        with torch.no_grad():
            grid_tensor = torch.FloatTensor(state['map_grid']).unsqueeze(0).to(self.device)
            metrics_tensor = torch.FloatTensor(state['metrics']).unsqueeze(0).to(self.device)
            
            # Sieć zwraca przewidywaną nagrodę dla wszystkich 15001 akcji
            q_values = self.policy_net(grid_tensor, metrics_tensor)
            
            # --- ACTION MASKING ---
            masked_q_values = q_values.clone()
            
            for action_id in range(self.action_size - 1): # Pomijamy No-Op
                # Agent pyta środowisko (env) o dekodowanie akcji
                x, z, _, _ = env._decode_action(action_id)
                
                # Jeśli pole jest zajęte, agent "zabija" tę akcję przypisując jej -Nieskończoność
                if state['map_grid'][x][z] != 0:
                    masked_q_values[0][action_id] = -float('inf')
            
            # Wybieramy najlepszą spośród w 100% legalnych akcji
            return masked_q_values.argmax().item()


    def optimize_model(self):
        if len(self.memory) < self.batch_size: 
            return
            
        # ZMIANA 1: Pobieramy batch, indeksy z drzewa oraz wagi IS
        batch, tree_indices, is_weights = self.memory.sample(self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        state_grids = torch.FloatTensor(np.array([s['map_grid'] for s in states])).to(self.device)
        state_metrics = torch.FloatTensor(np.array([s['metrics'] for s in states])).to(self.device)
        next_grids = torch.FloatTensor(np.array([s['map_grid'] for s in next_states])).to(self.device)
        next_metrics = torch.FloatTensor(np.array([s['metrics'] for s in next_states])).to(self.device)
        
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device) # Wagi na tensor
        
        # 1. Bieżące Q-values
        q_values = self.policy_net(state_grids, state_metrics).gather(1, actions)
        
        with torch.no_grad():
            # Double DQN: Wybór akcji główną siecią, ocena siecią docelową
            next_actions = self.policy_net(next_grids, next_metrics).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_grids, next_metrics).gather(1, next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
        # ZMIANA 2: Wyliczanie błędu TD dla każdego elementu w batchu (zwraca wektor rozmiaru 128)
        # Używamy reduction='none', by zważyć każdy błąd osobno, zanim wyciągniemy średnią
        td_errors = target_q_values - q_values
        loss = (is_weights * nn.SmoothL1Loss(reduction='none')(q_values, target_q_values)).mean()
        
        # ZMIANA 3: Propagacja wsteczna (bez zmian)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # ZMIANA 4: Aktualizacja priorytetów w drzewie SumTree
        # Wyciągamy czyste wartości błędu TD na procesor (CPU), by zaktualizować pamięć w Numpy
        numpy_td_errors = td_errors.detach().cpu().numpy().flatten()
        self.memory.update_priorities(tree_indices, numpy_td_errors)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_agent():
    env = CityEnv(grid_size=20)
    agent = DQNAgent(grid_size=20, metrics_size=11, action_size=env.action_space_size)
    
    # Hiperparametry treningu
    EPISODES = 30
    STEPS_PER_EPISODE = 60
    TARGET_UPDATE_FREQ = 5 # Co ile epizodów odświeżać sieć docelową
    
    try:
        print("Uruchamianie środowiska...")
        state, _, _ = env.connect()
        
        for episode in range(EPISODES):
            print(f"\n--- Rozpoczęcie Epizodu {episode + 1}/{EPISODES} ---")
            total_reward = 0
            state = env.reset()
            time.sleep(10)

            for step in range(STEPS_PER_EPISODE):
                # 1. Agent wybiera akcję na podstawie stanu
                action = agent.select_action(state, env)
                
                # 2. Wykonanie akcji w grze
                next_state, reward, done = env.step(action)
                
                # 3. Zapisanie do bufora powtórek
                agent.memory.push(state, action, reward, next_state, done)
                
                # 4. Nauka (jeśli w buforze jest wystarczająco dużo danych)
                agent.optimize_model()
                
                state = next_state
                total_reward += reward
                print(f"Stan świata: \nPopulacja: {state['metrics'][0]}, Szczęście: {state['metrics'][1]}, Średnia długość życia: {state['metrics'][2]}, Bezrobocie: {state['metrics'][3]}, Dochód: {state['metrics'][4]}, Wydatki: {state['metrics'][5]}, Zanieczyszczenie wody: {state['metrics'][6]}, Zanieczyszczenie gleby: {state['metrics'][7]}, Popyt na mieszkania: {state['metrics'][8]}, Popyt na handel: {state['metrics'][9]}, Popyt na miejsca pracy: {state['metrics'][10]}")
                print(f"Krok: {step+1} | Akcja: {action} | Nagroda: {reward:.2f} | Epsilon: {agent.epsilon:.2f}\n")
                
                # Opóźnienie, aby dać silnikowi Unity czas na przetworzenie symulacji
                time.sleep(3) 
                
            # Zmniejszanie eksploracji (Epsilon Decay) pod koniec każdego epizodu
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
                
            # Aktualizacja sieci docelowej
            if episode % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
                print(">> Zaktualizowano sieć docelową (Target Network)")
                
            print(f"Suma nagród w epizodzie: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\nPrzerwano trening ręcznie.")
    finally:
        env.close()
        torch.save(agent.policy_net.state_dict(), f"cities_skylines_rainbow_dqn_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pth")
        print("Zapisano model do cities_skylines_rainbow_dqn.pth")

if __name__ == '__main__':
    train_agent()