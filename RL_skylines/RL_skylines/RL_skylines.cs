using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ICities;
using UnityEngine;
using ColossalFramework;
using ColossalFramework.Plugins;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.IO;


namespace RL_skylines
{
    public class RL_skylines : IUserMod
    {
        public string Name
        {
            get { return "My RL Agent :)"; }
        }

        public string Description
        {
            get { return "Mod for Agent to actually work"; }
        }
    }

    // Ta klasa odpowiada za moment załadowania poziomu (miasta)
    public class ModLoadingExtension : LoadingExtensionBase
    {
        private GameObject _modGameObject;

        public override void OnLevelLoaded(LoadMode mode)
        {
            if (mode != LoadMode.NewGame && mode != LoadMode.LoadGame)
                return;

            // Tworzymy nowy pusty obiekt w grze i podpinamy do niego nasz skrypt
            _modGameObject = new GameObject("RL_Agent_Controller");
            _modGameObject.AddComponent<DataReader>();
        }

        public override void OnLevelUnloading()
        {
            if (_modGameObject != null)
            {
                GameObject.Destroy(_modGameObject);
            }
        }
    }

    public class PerformanceMeasures
    {
        // Deklaracja przykładowych metryk
        public uint finalPopulation;
        public int finalHappiness;
        public int actualResidentialDemand;
        public long totalIncome;
        public long currentMoneyAmount;

        public void get_performance_measures()
        {
            // Populacja i zadowolenie
            finalPopulation = DistrictManager.instance.m_districts.m_buffer[0].m_populationData.m_finalCount;
            finalHappiness = DistrictManager.instance.m_districts.m_buffer[0].m_finalHappiness;
            actualResidentialDemand = ZoneManager.instance.m_actualResidentialDemand;

            // Te są złe do zmiany!!!
            totalIncome = EconomyManager.playerMoney;
            currentMoneyAmount = EconomyManager.instance.m_EconomyWrapper.currentMoneyAmount;

        }

        public void print_performance_measures()
        {
            string output = $"Pop: {finalPopulation}, Hap: {finalHappiness}, ResDem: {actualResidentialDemand}, Inc: {totalIncome}, Exp: {currentMoneyAmount}";

            // Wypisanie bezpośrednio do konsoli F7 w grze:
            DebugOutputPanel.AddMessage(
                PluginManager.MessageType.Message,
                "[RL_Skylines] Metryki: " + output
            );
        }

        public string performance_measures_cs()
        {
            // Zwraca string oddzielony przecinkami do przesłania potokiem (Named Pipes) do Pythona
            return $"{finalPopulation},{finalHappiness},{actualResidentialDemand},{totalIncome},{currentMoneyAmount}`";
        }
    }

    // 4. Główna pętla i kontroler w grze
    public class DataReader : MonoBehaviour
    {
        Thread thread;
        public int connectionPort = 25001;
        TcpListener server;
        TcpClient client;
        bool running;

        // Nasze zmienne do metryk
        private PerformanceMeasures metrics;
        private string latestData = "0,0,0,0,0";
        private float timer = 7.0f;
        private float actionInterval = 7.0f;

        // Bezpieczne przekazywanie logów do wątku głównego (Unity nie lubi Debug.Log w innych wątkach)
        private string threadLogMessage = null;
        private string lastAction = null;

        void Start()
        {
            metrics = new PerformanceMeasures();
            DebugOutputPanel.AddMessage(
                PluginManager.MessageType.Message,
                "[RL_Skylines] Start serwera TCP (Byte Buffer)..."
            );

            // ExportPrefabList();

            // Uruchamiamy wątek sieciowy
            ThreadStart ts = new ThreadStart(GetData);
            thread = new Thread(ts);
            thread.IsBackground = true;
            thread.Start();


        }

        void Update()
        {
            if (threadLogMessage != null)
            {
                DebugOutputPanel.AddMessage(
                    PluginManager.MessageType.Message,
                    threadLogMessage);
                threadLogMessage = null;
            }

            // 1. Sprawdzamy, czy w tle odebrano nową akcję
            if (lastAction != null)
            {
                DebugOutputPanel.AddMessage(
                    PluginManager.MessageType.Message,
                    "[RL_Skylines] Przystępuję do wykonania akcji: " + lastAction);

                // Wykonujemy akcję w głównym wątku gry!
                ExecuteAction(lastAction);

                lastAction = null; // Czyścimy komendę po wykonaniu
            }

            // 2. Aktualizowanie metryk co 7 sekund w tle
            timer += Time.deltaTime;
            if (timer >= actionInterval)
            {
                timer -= actionInterval;
                try
                {
                    metrics.get_performance_measures();
                    latestData = metrics.performance_measures_cs();
                }
                catch (System.Exception) { }
            }
        }

        // Główny dekoder komend z Pythona
        void ExecuteAction(string command)
        {
            if (string.IsNullOrEmpty(command)) return;

            try
            {
                string[] tokens = command.Split(' ');
                string actionType = tokens[0].ToLower();

                switch (actionType)
                {
                    case "createbuilding":
                        // Parsowanie argumentów: createbuilding <prefab_id> <x> <z>
                        uint prefabId = uint.Parse(tokens[1]);
                        float bx = float.Parse(tokens[2]);
                        float bz = float.Parse(tokens[3]);

                        Vector3 buildPosition = new Vector3(bx, 0, bz);

                        // 1. Pobieramy projekt budynku (Prefab) z gry na podstawie ID
                        BuildingInfo prefab = PrefabCollection<BuildingInfo>.GetPrefab(prefabId);

                        if (prefab == null)
                        {
                            DebugOutputPanel.AddMessage(
                                PluginManager.MessageType.Warning,
                                $"[RL_Skylines] Nie znaleziono budynku o ID: {prefabId}");
                            return;
                        }

                        // 2. Gra wymaga obiektu Randomizer do generowania wariantów wizualnych
                        ColossalFramework.Math.Randomizer randomizer = new ColossalFramework.Math.Randomizer(1337);

                        // 3. Stawiamy budynek w świecie gry
                        ushort newBuildingId;
                        BuildingManager.instance.CreateBuilding(
                            out newBuildingId,
                            ref randomizer,
                            prefab,
                            buildPosition,
                            0f, // Kąt obrotu (angle)
                            0,  // Długość (hitbox length) - zazwyczaj 0 dla punktowych budynków
                            SimulationManager.instance.m_currentBuildIndex
                        );

                        DebugOutputPanel.AddMessage(
                            PluginManager.MessageType.Message,
                            $"[RL_Skylines] Zbudowano: {prefab.name} (ID: {newBuildingId}) na pozycji {buildPosition}");
                        break;

                    case "createzone":
                        // Zgodnie z pracą: createzone <typ> <x> <z> <ilość_bloków>
                        int zoneType = int.Parse(tokens[1]);
                        float zx = float.Parse(tokens[2]);
                        float zz = float.Parse(tokens[3]);
                        int quantity = 1; // Domyślnie zmieniamy 1 blok
                        if (tokens.Length > 4) quantity = int.Parse(tokens[4]);

                        Vector3 zonePosition = new Vector3(zx, 0, zz);

                        // Wywołujemy naszą nową funkcję strefującą
                        int zonedAmount = ZoneArea(zonePosition, zoneType, quantity);

                        DebugOutputPanel.AddMessage(
                            PluginManager.MessageType.Message,
                            $"[RL_Skylines] Strefowanie: nałożono strefę typu {zoneType} na {zonedAmount} komórkach przy {zonePosition}");
                        break;


                    default:
                        DebugOutputPanel.AddMessage(
                            PluginManager.MessageType.Warning,
                            $"[RL_Skylines] Nieznana komenda: {actionType}");
                        break;
                }
            }
            catch (System.Exception e)
            {
                DebugOutputPanel.AddMessage(
                    PluginManager.MessageType.Error,
                    "[RL_Skylines] Błąd parsowania komendy: " + e.Message);
            }
        }

        // Funkcja odtwarzająca zachowanie z pracy dyplomowej
        private int ZoneArea(Vector3 position, int zoneType, int quantity)
        {
            int zonedCount = 0;
            ZoneManager zm = ZoneManager.instance;
            ItemClass.Zone type = (ItemClass.Zone)zoneType;

            // Szukamy najbliższego bloku stref wygenerowanego przez drogi
            ushort closestBlock = 0;
            float closestDist = 5000f; // Limit z pracy dyplomowej

            // Iterujemy po wszystkich istniejących blokach stref w grze
            for (ushort i = 1; i < zm.m_blocks.m_buffer.Length; i++)
            {
                // Sprawdzamy, czy blok w ogóle istnieje (flaga Created)
                if ((zm.m_blocks.m_buffer[i].m_flags & ZoneBlock.FLAG_CREATED) != 0)
                {
                    float dist = Vector3.Distance(position, zm.m_blocks.m_buffer[i].m_position);

                    // Jeśli znajdziemy bliższy blok, zapisujemy go
                    if (dist < closestDist)
                    {
                        closestDist = dist;
                        closestBlock = i;
                    }
                }
            }

            // Jeśli znaleźliśmy jakikolwiek blok blisko koordynatów
            if (closestBlock != 0)
            {
                // C:S przechowuje strefy w sposób bardzo skomplikowany (jako bity w zmiennych m_zone1 i m_zone2).
                // Dla potrzeb algorytmu RL upraszczamy to: pokrywamy cały znaleziony blok wybraną strefą.
                ulong mask = GenerateZoneMask((ulong)type);

                zm.m_blocks.m_buffer[closestBlock].m_zone1 = mask;
                zm.m_blocks.m_buffer[closestBlock].m_zone2 = mask;

                // Odświeżamy siatkę, aby gra zauważyła zmianę i zaczęła budować domy
                zm.m_blocks.m_buffer[closestBlock].UpdateBlock(closestBlock);

                zonedCount += 16; // Zaznaczyliśmy cały blok (który składa się z 32 małych kafelków) zmienione na 16
            }

            return zonedCount;
        }

        // Funkcja pomocnicza generująca maskę bitową dla silnika gry
        private ulong GenerateZoneMask(ulong typeId)
        {
            ulong mask = 0;
            // Blok strefy to 16 kafelków w m_zone1 i 16 w m_zone2. 
            // Każdy kafelek zajmuje 4 bity (16 * 4 = 64 bity = 1 ulong).
            for (int i = 0; i < 16; i++)
            {
                mask |= (typeId << (i * 4));
            }
            return mask;
        }
        void GetData()
        {
            try
            {
                // Używamy IPAddress.Any tak jak w Twoim przykładzie
                server = new TcpListener(IPAddress.Any, connectionPort);
                server.Start();
                threadLogMessage = "[RL_Skylines] Serwer czeka na port 25001...";

                client = server.AcceptTcpClient();
                threadLogMessage = "[RL_Skylines] Python połączony!";

                running = true;
                while (running)
                {
                    Connection();
                }
                server.Stop();
            }
            catch (System.Exception e)
            {
                threadLogMessage = "[RL_Skylines] BŁĄD TCP: " + e.Message;
            }
        }

        void Connection()
        {
            NetworkStream nwStream = client.GetStream();
            byte[] buffer = new byte[client.ReceiveBufferSize];

            // Gra zatrzymuje się w tym miejscu i czeka na wiadomość od Pythona
            int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize);

            if (bytesRead > 0)
            {
                // Dekodujemy komendę od Pythona (np. "createzone")
                string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                lastAction = dataReceived.Trim();

                // Odpowiadamy aktualnym stanem metryk
                byte[] stateBuffer = Encoding.UTF8.GetBytes(latestData + "\n");
                nwStream.Write(stateBuffer, 0, stateBuffer.Length);
            }
            else
            {
                // Zabezpieczenie przed rozłączeniem
                running = false;
            }
        }

        private void ExportPrefabList()
        {
            try
            {
                // Plik zapisze się na Twoim pulpicie
                string desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
                string filePath = Path.Combine(desktopPath, "CS_Prefabs_List.txt");

                using (StreamWriter sw = new StreamWriter(filePath))
                {
                    sw.WriteLine("--- LISTA BUDYNKÓW CITIES: SKYLINES ---");

                    // Pobieramy liczbę wszystkich załadowanych budynków
                    int prefabCount = PrefabCollection<BuildingInfo>.LoadedCount();

                    for (uint i = 0; i < prefabCount; i++)
                    {
                        BuildingInfo info = PrefabCollection<BuildingInfo>.GetPrefab(i);

                        // Filtrujemy tylko realne budynki, które gracz/agent może postawić (usługi, infrastruktura)
                        if (info != null && info.m_class != null && info.m_placementStyle == ItemClass.Placement.Manual)
                        {
                            // Zapisujemy ID, systemową nazwę oraz kategorię (np. Electricity, HealthCare)
                            sw.WriteLine($"ID: {i} | Kategoria: {info.m_class.m_service} | Nazwa: {info.name}");
                        }
                    }
                }

                DebugOutputPanel.AddMessage(
                    PluginManager.MessageType.Message,
                    "[RL_Skylines] Pomyślnie wyeksportowano listę prefabów na Pulpit!");
            }
            catch (Exception e)
            {
                DebugOutputPanel.AddMessage(
                    PluginManager.MessageType.Error,
                    "[RL_Skylines] Błąd eksportu: " + e.Message);
            }
        }

        void OnDestroy()
        {
            running = false;
            if (server != null) server.Stop();
        }
    }
}