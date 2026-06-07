import time
from city_env import CityEnv
import numpy as np

BUILDING_CATALOG = {
    "water_tower": 41,
    "coal_power_plant": 379,
    "wind_turbine": 374,
    "medical_clinic": 655,
    "regular_park": 292
}

# --- TESTOWANIE ŚRODOWISKA ---
# --- TESTOWANIE ŚRODOWISKA ---
if __name__ == '__main__':
    env = CityEnv()
    
    try:
        # ROZWIĄZANIE: Rozpakowujemy krotkę. Ignorujemy nagrodę i flagę done za pomocą "_"
        initial_state, _, _ = env.connect() 
        print("\n=== START ===")
        print(f"Metryki początkowe: {initial_state['metrics']}")
        
        # Symulujemy 5 kroków losowego agenta
        for step_num in range(1, 50):
            print(f"\nKrok {step_num}:")
            
            random_action = np.random.randint(0, env.action_space_size)
            
            # ROZWIĄZANIE: Tutaj również odbieramy pełną trójkę danych
            state, reward, done = env.step(random_action)
            
            print(f"Agent wykonał akcję ID: {random_action}")
            print(f"Otrzymana nagroda (Reward): {reward}")
            print(f"Populacja: {state['metrics'][0]}, Szczęście: {state['metrics'][1]}, Średnia długość życia: {state['metrics'][2]}, Bezrobocie: {state['metrics'][3]}, Dochód: {state['metrics'][4]}, Wydatki: {state['metrics'][5]}, Zanieczyszczenie wody: {state['metrics'][6]}, Zanieczyszczenie gleby: {state['metrics'][7]}, Popyt na mieszkania: {state['metrics'][8]}, Popyt na handel: {state['metrics'][9]}, Popyt na miejsca pracy: {state['metrics'][10]}")
            print(f"Aktualne metryki: {state['metrics']}")
            print("Wycinek mapy (górny lewy róg 3x3):")
            print(state['map_grid'][:3, :3])
            
            time.sleep(7) 
            
    except KeyboardInterrupt:
        print("\nZakończono przez użytkownika.")
    finally:
        env.close()