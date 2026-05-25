import socket
import time
import numpy as np
import re

class CityEnv:
    def __init__(self, host='127.0.0.1', port=25001, grid_size=10):
        self.host = host
        self.port = port
        self.grid_size = grid_size
        
        # Słownik dostępnych stref: 2=ResLow, 4=ComLow, 6=Ind
        self.zone_types = [2, 4, 6] 
        
        # Przestrzeń akcji: (10 * 10 komórek) * 3 typy stref = 300 możliwych akcji
        self.action_space_size = (self.grid_size ** 2) * len(self.zone_types)
        
        # Stan wewnętrzny (Macierz mapy 10x10)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Wektor metryk z C# (Populacja, Zadowolenie, Popyt, Przychód, Wydatki)
        self.metrics = np.zeros(5, dtype=np.float32)
        
        self.sock = None
        self.pipe = None

        # Wektor metryk z C# (Populacja, Zadowolenie, Popyt, Przychód, Wydatki)
        self.metrics = np.zeros(5, dtype=np.float32)
        self.previous_metrics = np.zeros(5, dtype=np.float32) # NOWE

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
        """Wykonuje akcję w grze i zwraca nowy stan."""

        # Zapisujemy stan przed wykonaniem akcji
        self.previous_metrics = self.metrics.copy()

        if action_id is not None:
            # 1. Dekodujemy ID akcji z sieci neuronowej na X, Z i Typ Strefy
            x, z, zone_type = self._decode_action(action_id)
            
            # 2. Aktualizujemy naszą wewnętrzną "mapę cienia" w NumPy
            self.grid[x][z] = zone_type
            
            # 3. Formatujemy komendę dla gry (np. "createzone 2 50 120 1")
            # Skalujemy x i z (np. mnożąc przez 10), aby strefy nie nakładały się na siebie
            world_x, world_z = x * 10, z * 10
            command = f"createzone {zone_type} {world_x} {world_z} 1"
            
            # 4. Wysyłamy komendę do C#
            self.sock.sendall((command + '\n').encode("utf-8"))
        else:
            # Pusta akcja (inicjalizacja)
            self.sock.sendall("ping\n".encode("utf-8"))

        # 5. Odbieramy najświeższe metryki od gry
        response = self.pipe.readline().strip()
        if not response:
            raise ConnectionError("Gra zerwała połączenie.")
        
        # 6. Aktualizujemy wektor metryk
        self._update_metrics(response)
        
        # 7. Obliczamy nagrodę za ten krok!
        reward = self._calculate_reward()

        # Zwracamy kompletny stan, nagrodę i flagę czy epizod się skończył
        return self._get_state(), reward, False

    def _decode_action(self, action_id):
        """
        Matematyka dekodowania:
        Każda komórka ma 3 możliwe akcje. 
        Dzieląc action_id, uzyskujemy współrzędne i typ strefy.
        """
        zone_idx = action_id % len(self.zone_types)
        cell_idx = action_id // len(self.zone_types)
        
        x = cell_idx // self.grid_size
        z = cell_idx % self.grid_size
        
        return x, z, self.zone_types[zone_idx]

    def _update_metrics(self, response_str):
        """Zamienia string na tablicę NumPy z czyszczeniem danych."""
        try:
            values = response_str.split(',')
            clean_values = []
            
            for v in values:
                # Wyciągamy ze stringa WYŁĄCZNIE cyfry, minus i kropkę
                # Usunie to wszelkie backticki (`), litery czy białe znaki
                clean_v = re.sub(r'[^\d.-]', '', v)
                
                # Zabezpieczenie, gdyby po czyszczeniu string okazał się pusty
                if clean_v == '':
                    clean_v = '0'
                    
                clean_values.append(clean_v)
                
            if len(clean_values) >= 5:
                self.metrics = np.array(clean_values[:5], dtype=np.float32)
                
        except Exception as e:
            # W razie absolutnej awarii wypisze nam dokładną, "surową" zawartość stringa
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
        # Indeksy metryk: 0=Pop, 1=Hap, 2=ResDem, 3=Inc, 4=Exp
        
        delta_population = self.metrics[0] - self.previous_metrics[0]
        delta_happiness = self.metrics[1] - self.previous_metrics[1]
        
        profit = self.metrics[3] - self.metrics[4] # Przychód minus wydatki
        
        # Konstruujemy sygnał nagrody. Wagi będą wzięte z pracy.
        reward = (delta_population * 0.5) + (delta_happiness * 2.0)
        
        # Dodajemy karę za ujemny bilans finansowy miasta
        if profit < 0:
            reward -= 10.0
            
        # Opcjonalnie: stała kara za każdy krok (time penalty), 
        # wymuszająca na agencie szybsze działanie
        reward -= 0.1 
        
        return float(reward)
