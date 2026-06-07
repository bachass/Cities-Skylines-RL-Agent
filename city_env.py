import socket
import time
import numpy as np
import re

class CityEnv:
    def __init__(self, host='127.0.0.1', port=25001, grid_size=50):
        self.host = host
        self.port = port
        self.grid_size = grid_size
        self.world_step_size = 8.0 
        
        # KATALOG AKCJI: Co agent może postawić na pojedynczym kafelku?
        # Format: ("typ_komendy_w_C#", ID_w_grze)
        self.action_catalog = [
            ("zone", 2),         # 0: Strefa Mieszkalna (ResLow)
            ("zone", 4),         # 1: Strefa Komercyjna (ComLow)
            ("zone", 6),         # 2: Strefa Przemysłowa (Ind)
        ]
        
        # Przestrzeń akcji: (50 * 50 komórek * 6 narzędzi) + 1 (Nic nie rób) = 15001 akcji
        self.action_space_size = (self.grid_size ** 2) * len(self.action_catalog) + 1
        
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        self.metrics = np.zeros(11, dtype=np.float32)
        self.previous_metrics = np.zeros(11, dtype=np.float32)
        
        # Flaga do karania agenta za hakowanie nagrody
        self.illegal_move_penalty = False 
        
        self.sock = None
        self.pipe = None


    def connect(self):
        """Nawiązuje połączenie TCP z silnikiem gry."""
        print(f"Łączenie z grą na {self.host}:{self.port}...")
        while True:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                self.pipe = self.sock.makefile('rw', buffering=1)
                print("Połączono pomyślnie!")
                
                # Pobieramy stan początkowy, wymuszając "pustą" akcję
                return self.step(None) 
            except ConnectionRefusedError:
                time.sleep(2)

    def step(self, action_id):
        self.previous_metrics = self.metrics.copy()
        self.illegal_move_penalty = False # Resetujemy flagę kary

        if action_id is not None:
            # 1. Odkodowujemy akcję
            x, z, command_type, game_id = self._decode_action(action_id)
            
            # 2. Obsługa akcji "Do Nothing" (Ostatnie ID w przestrzeni)
            if command_type == "noop":
                self.sock.sendall("ping\n".encode("utf-8"))
                
            else:
                # 3. SPRAWDZANIE HAKOWANIA (Czy kafelek jest już zajęty?)
                if self.grid[x][z] != 0:
                    # Agent próbuje nadpisać kafelek! Nakładamy flagę kary i marnujemy ruch.
                    self.illegal_move_penalty = True
                    self.sock.sendall("ping\n".encode("utf-8"))
                else:
                    # Legalny ruch! Aktualizujemy wirtualną mapę
                    self.grid[x][z] = game_id
                    
                    world_x = x * self.world_step_size
                    world_z = z * self.world_step_size 
                    
                    # 4. Wysyłanie dynamicznej komendy (Strefa vs Budynek)
                    if command_type == "zone":
                        command = f"createzone {game_id} {world_x} {world_z} 1"
                    elif command_type == "building":
                        command = f"createbuilding {game_id} {world_x} {world_z}"
                        
                    self.sock.sendall((command + '\n').encode("utf-8"))
        else:
            self.sock.sendall("ping\n".encode("utf-8"))

        response = self.pipe.readline().strip()
        if not response:
            raise ConnectionError("Gra zerwała połączenie.")
        
        self._update_metrics(response)
        reward = self._calculate_reward()
        
        return self._get_state(), reward, False

    def _decode_action(self, action_id):
        """Nowy system dekodowania włączający budynki i No-Op"""
        # Jeśli to ostatni dostępny indeks, agent zdecydował się przeczekać turę
        if action_id == self.action_space_size - 1:
            return None, None, "noop", 0
            
        action_idx = action_id % len(self.action_catalog)
        cell_idx = action_id // len(self.action_catalog)
        
        x = cell_idx // self.grid_size
        z = cell_idx % self.grid_size
        
        command_type, game_id = self.action_catalog[action_idx]
        return x, z, command_type, game_id

    def _update_metrics(self, response_str):
        """Aktualizacja z nowym rozmiarem tablicy."""
        try:
            values = response_str.split(',')
            clean_values = []
            for v in values:
                clean_v = re.sub(r'[^\d.-]', '', v)
                if clean_v == '': clean_v = '0'
                clean_values.append(clean_v)
                
            # ZMIANA 2: Oczekujemy 11 wartości
            if len(clean_values) >= 11:
                self.metrics = np.array(clean_values[:11], dtype=np.float32)
        except Exception as e:
            print(f"[!] BŁĄD DEKODOWANIA METRYK! Surowy string z C#: {repr(response_str)}")
            raise e

    def _get_state(self):
        """Pakuje stan środowiska dla sieci neuronowej."""
        return {
            "map_grid": self.grid.copy(),
            "metrics": self.metrics.copy()
        }

    def close(self):
        if self.sock:
            self.sock.close()

    def _calculate_reward(self):
        """Nowa funkcja nagrody oparta na macierzy wag."""
        
        # Obliczamy deltę (zmianę) dla każdej z 11 metryk jednocześnie
        deltas = self.metrics - self.previous_metrics
        
        # Definiujemy wagi dla każdego parametru (Indeksy od 0 do 10)
        # UWAGA: Wagi dostosuj według własnych upodobań treningowych!
        weights = np.array([
            10.0,    # 0: Pop (Wzrost populacji to plus)
            3.0,    # 1: Hap (Zadowolenie jest bardzo ważne)
            1.0,    # 2: AvgLife (Wzrost dł. życia to plus)
            -2.0,   # 3: Unemp (Spadek bezrobocia to plus)
            0.001,   # 4: Inc (Pieniądze często rosną w tysiącach, więc mniejsza waga)
            -0.0001,  # 5: Exp (Wzrost wydatków to minus)
            -1.0,   # 6: WaterPol (Zanieczyszczenia to surowa kara)
            -1.0,   # 7: GroundPol (Zanieczyszczenia to surowa kara)
            -5.0,   # 8: ResDem (Spadek zapotrzebowania to plus - spełniono potrzebę!)
            -5.0,   # 9: ComDem
            -5.0    # 10: WorkDem
        ], dtype=np.float32)
        
        # Magia NumPy: Iloczyn skalarny. Mnoży każdą deltę przez jej wagę i sumuje wszystko w jedną liczbę.
        reward = np.dot(deltas, weights)            
        reward -= 0.1 # Time penalty (kara za upływający czas)

        if self.illegal_move_penalty:
            # Bardzo bolesna kara. Agent szybko oduczy się stawiania tam, gdzie już coś jest.
            reward -= 50.0
        
        return float(reward)
