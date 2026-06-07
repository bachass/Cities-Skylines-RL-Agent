import torch
import numpy as np
import time

# Importujemy nasze zaktualizowane środowisko i architekturę sieci
from city_env import CityEnv
from dqn_agent import QNetwork 

def run_trained_agent(model_path="cities_skylines_dqn_2026-06-07_15-43-31.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    # Konfiguracja środowiska na siatkę 50x50
    env = CityEnv(grid_size=50)
    action_size = env.action_space_size
    
    # 1. Inicjalizacja sieci neuronowej
    policy_net = QNetwork(grid_size=50, metrics_size=11, action_size=action_size).to(device)
    
    # 2. Załadowanie wyuczonych wag
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Pomyślnie załadowano wagi z pliku {model_path}!")
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku {model_path}.")
        return
        
    policy_net.eval()

    # Słowniki pomocnicze do wyświetlania ładnych logów w konsoli
    zone_names = {2: "Mieszkalna", 4: "Komercyjna", 6: "Przemysłowa"}
    building_names = {5: "Elektrownia/Wiatrak", 12: "Pompa Wody", 17: "Klinika Zdrowia"}

    try:
        print("\nOczekiwanie na Cities: Skylines...")
        state, _, _ = env.connect()
        print("Gra podłączona! Agent rozpoczyna architekturę miasta.")
        
        step = 1
        while True:
            with torch.no_grad():
                grid_tensor = torch.FloatTensor(state['map_grid']).unsqueeze(0).to(device)
                metrics_tensor = torch.FloatTensor(state['metrics']).unsqueeze(0).to(device)
                
                # Sieć ocenia wszystkie 15001 możliwych akcji
                q_values = policy_net(grid_tensor, metrics_tensor)
                
                # --- ACTION MASKING DLA INFERENCJI ---
                # Wykluczamy z puli decyzyjnej kafelki, na których już coś stoi
                masked_q_values = q_values.clone()
                for action_id in range(action_size - 1): # Pomijamy No-Op, bo zawsze wolno czekać
                    x, z, _, _ = env._decode_action(action_id)
                    if state['map_grid'][x][z] != 0:
                        masked_q_values[0][action_id] = -float('inf')
                
                # Agent wybiera najlepszą, w 100% legalną akcję
                best_action = masked_q_values.argmax().item()
            
            # --- DEKODOWANIE DLA LOGÓW ---
            x, z, command_type, game_id = env._decode_action(best_action)
            
            if command_type == "noop":
                action_desc = "Nic nie robię (Oczekiwanie na budżet/zmianę metryk)"
            elif command_type == "zone":
                action_desc = f"Strefa: {zone_names.get(game_id, 'Nieznana')} na ({x}, {z})"
            elif command_type == "building":
                action_desc = f"Budynek: {building_names.get(game_id, 'Nieznany')} na ({x}, {z})"
            else:
                action_desc = "Nieznana akcja"

            # Wykonanie akcji w grze
            next_state, reward, done = env.step(best_action)
            
            print(f"Krok {step}: {action_desc} | Przewidziana nagroda (Q-Value): {reward:.2f}")
            
            state = next_state
            step += 1
            
            time.sleep(1) 

    except KeyboardInterrupt:
        print("\nAgent zatrzymany przez użytkownika.")
    finally:
        env.close()

if __name__ == '__main__':
    run_trained_agent()