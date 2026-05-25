import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from city_env import CityEnv
import time

# --- 1. ARCHITEKTURA SIECI NEURONOWEJ ---
class QNetwork(nn.Module):
    def __init__(self, grid_size=10, metrics_size=5, action_size=300):
        super(QNetwork, self).__init__()
        
        # 100 (z siatki) + 5 (z metryk) = 105 neuronów na wejściu
        input_dim = (grid_size * grid_size) + metrics_size
        
        # Prosta, ale głęboka sieć wielowarstwowa (MLP)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        
    def forward(self, grid, metrics):
        # Spłaszczanie mapy: z wymiaru (batch, 10, 10) na (batch, 100)
        grid_flat = grid.view(grid.size(0), -1)
        
        # Łączenie mapy i metryk w jeden tensor
        x = torch.cat((grid_flat, metrics), dim=1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Zwracamy surowe wartości Q dla wszystkich 300 akcji
        return self.fc3(x)

# --- 2. REPLAY BUFFER (DATA LAKE DLA RL) ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        # Stan to u nas słownik, musimy go przechować w całości
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)
    
# --- 3. KLASA AGENTA ---
class DQNAgent:
    def __init__(self, grid_size=10, metrics_size=5, action_size=300):
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inicjalizacja dwóch sieci
        self.policy_net = QNetwork(grid_size, metrics_size, action_size).to(self.device)
        self.target_net = QNetwork(grid_size, metrics_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Sieć docelowa nie jest trenowana przez backprop
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayBuffer(capacity=10000)
        
        # Hiperparametry
        self.batch_size = 64
        self.gamma = 0.99       # Czynnik dyskontujący przyszłe nagrody
        self.epsilon = 1.0      # Zaczynamy od 100% losowych akcji (Eksploracja)
        self.epsilon_min = 0.1  # Minimalny próg losowości
        self.epsilon_decay = 0.995 # Jak szybko agent przestaje eksplorować
        
    def select_action(self, state):
        """Strategia Epsilon-Greedy: wybór między eksploracją a eksploatacją."""
        if np.random.rand() <= self.epsilon:
            # Losowa akcja
            return random.randrange(self.action_size)
            
        # Akcja z sieci (Eksploatacja)
        with torch.no_grad():
            # Konwersja numpy array na tensory PyTorcha
            grid_tensor = torch.FloatTensor(state['map_grid']).unsqueeze(0).to(self.device)
            metrics_tensor = torch.FloatTensor(state['metrics']).unsqueeze(0).to(self.device)
            
            # Przepuszczamy stan przez sieć
            q_values = self.policy_net(grid_tensor, metrics_tensor)
            
            # Zwracamy indeks akcji o najwyższej wartości Q
            return q_values.argmax().item()

    def optimize_model(self):
        """Pobiera batch danych z bufora i aktualizuje wagi sieci."""
        if len(self.memory) < self.batch_size:
            return # Nie trenujemy, dopóki nie zbierzemy wystarczającej ilości danych
            
        batch = self.memory.sample(self.batch_size)
        
        # Rozpakowywanie batcha
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Przygotowanie tensorów wejściowych ze słowników
        state_grids = torch.FloatTensor(np.array([s['map_grid'] for s in states])).to(self.device)
        state_metrics = torch.FloatTensor(np.array([s['metrics'] for s in states])).to(self.device)
        
        next_grids = torch.FloatTensor(np.array([s['map_grid'] for s in next_states])).to(self.device)
        next_metrics = torch.FloatTensor(np.array([s['metrics'] for s in next_states])).to(self.device)
        
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Krok 1: Obliczanie przewidywanej wartości Q dla wybranych akcji
        q_values = self.policy_net(state_grids, state_metrics).gather(1, actions)
        
        # Krok 2: Obliczanie docelowej wartości Q (Target) z użyciem zamrożonej sieci
        with torch.no_grad():
            max_next_q_values = self.target_net(next_grids, next_metrics).max(1)[0].unsqueeze(1)
            # Równanie Bellmana: Target = Reward + Gamma * Max(Next_Q) * (1 - Done)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
            
        # Krok 3: Obliczenie błędu (Huber Loss jest odporniejszy na szumy niż MSE)
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        # Krok 4: Propagacja wsteczna (Backpropagation)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Zapobieganie eksplodującym gradientom (Gradient Clipping)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        
        self.optimizer.step()
        
    def update_target_network(self):
        """Kopiuje aktualne wagi do sieci docelowej."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_agent():
    env = CityEnv()
    agent = DQNAgent(grid_size=10, metrics_size=5, action_size=env.action_space_size)
    
    # Hiperparametry treningu
    EPISODES = 50
    STEPS_PER_EPISODE = 10
    TARGET_UPDATE_FREQ = 5 # Co ile epizodów odświeżać sieć docelową
    
    try:
        print("Uruchamianie środowiska...")
        state, _, _ = env.connect()
        
        for episode in range(EPISODES):
            print(f"\n--- Rozpoczęcie Epizodu {episode + 1}/{EPISODES} ---")
            total_reward = 0
            
            for step in range(STEPS_PER_EPISODE):
                # 1. Agent wybiera akcję na podstawie stanu
                action = agent.select_action(state)
                
                # 2. Wykonanie akcji w grze
                next_state, reward, done = env.step(action)
                
                # 3. Zapisanie do bufora powtórek
                agent.memory.push(state, action, reward, next_state, done)
                
                # 4. Nauka (jeśli w buforze jest wystarczająco dużo danych)
                agent.optimize_model()
                
                state = next_state
                total_reward += reward
                print(f"Stan świata: {state['metrics']}")
                print(f"Krok: {step+1} | Akcja: {action} | Nagroda: {reward:.2f} | Epsilon: {agent.epsilon:.2f}\n")
                
                # Opóźnienie, aby dać silnikowi Unity czas na przetworzenie symulacji
                time.sleep(7) 
                
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
        # Zapisz wagi modelu na koniec!
        torch.save(agent.policy_net.state_dict(), "cities_skylines_dqn.pth")
        print("Zapisano model do cities_skylines_dqn.pth")

if __name__ == '__main__':
    train_agent()