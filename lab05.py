from PIL import Image
from collections import defaultdict
from abc import ABC, abstractmethod
from queue import PriorityQueue, Queue
from collections import deque
import math
from abc import ABC, abstractmethod

class ProblemFramework(ABC):
    def __init__(self, matrix):
        self.matrix = matrix

    @abstractmethod
    def actions(self, state):
        """
            Regresa las acciones que se pueden realizar en el estado
            actual.
        """
        pass

    @abstractmethod
    def step_cost(self, state, action, next_state):
        """
            Regresa el costo de realizar una acción en un estado.
        """
        pass

    @abstractmethod
    def goal_test(self, state):
        """
            Verfica si un estado es estado objetivo.
            Devuelve True si es estado objetivo, False en otro caso.
        """
        pass

    @abstractmethod
    def initial_state(self):
        """
            Regresa el estado inicial del problema.
        """
        pass

    @abstractmethod
    def result(self, state, action):
        """
            Regresa el estado resultante de realizar una acción en un estado.
        """
        pass

class Laberinto(ProblemFramework):
    def __init__(self, matrix):
        super().__init__(matrix)

    def step_cost(self, state, action, next_state):
    # Aquí define el costo de moverse de un estado a otro. Ejemplo:
        return 1  # Costo constante para cada movimiento.

    def actions(self, state):
        x, y = state

        acciones = []

        # Mover hacia arriba
        if y > 0 and self.matrix.get(y - 1).get(x) != (0, 0, 0):
            acciones.append("UP")

        # Mover hacia abajo
        if y < len(self.matrix) - 1 and self.matrix.get(y + 1).get(x) != (0, 0, 0):
            acciones.append("DOWN")

        # Mover hacia la izquierda
        if x > 0 and self.matrix.get(y).get(x - 1) != (0, 0, 0):
            acciones.append("LEFT")
        
        # Mover hacia la derecha
        if x < len(self.matrix.get(y)) - 1 and self.matrix.get(y).get(x + 1) != (0, 0, 0):
            acciones.append("RIGHT")
        
        return acciones

        
    def goal_test(self, state):
        return self.matrix.get(state[1]).get(state[0]) == (0, 255, 0)

    def initial_state(self):
        for y in self.matrix:
            for x in self.matrix.get(y):
                if self.matrix.get(y).get(x) == (255, 0, 0):
                    return (x, y)
    
    def result(self, state, action):
        x, y = state

        if action == "UP":
            return (x, y - 1)
        elif action == "DOWN":
            return (x, y + 1)
        elif action == "LEFT":
            return (x - 1, y)
        elif action == "RIGHT":
            return (x + 1, y)

class GraphSearch:
    def __init__(self, problem):
        self.problem = problem

    def printNewBMP(self, explorados, path, estadoAceptado):

        img = Image.new("RGB", (len(self.problem.matrix.get(0)), len(self.problem.matrix)), "white")
        for y in range(len(self.problem.matrix)):
            for x in range(len(self.problem.matrix.get(y))):
                if (x, y) in explorados:
                    img.putpixel((x, y), (0, 0, 255))
                else:
                    img.putpixel((x, y), self.problem.matrix.get(y).get(x))
                            
        img.save(path)

    def bfs(self):
        frontera = Queue()
        estadoInicial = self.problem.initial_state()
        frontera.put(estadoInicial)
        explorados = set()

        while not frontera.empty():
            estadoActual = frontera.get()
            if self.problem.goal_test(estadoActual):
                self.printNewBMP(explorados, "BFS.bmp", estadoActual)
                return estadoActual
            
            explorados.add(estadoActual)

            for action in self.problem.actions(estadoActual):
                siguienteEstado = self.problem.result(estadoActual, action)
                if siguienteEstado not in explorados and siguienteEstado not in frontera.queue:
                    frontera.put(siguienteEstado)

        return None

    def dfs(self):
        frontera = deque()
        estadoInicial = self.problem.initial_state()
        frontera.append(estadoInicial)
        explorados = set()

        while frontera:
            estadoActual = frontera.pop()
            if self.problem.goal_test(estadoActual):
                self.printNewBMP(explorados, "DFS.bmp", estadoActual)
                return estadoActual
            
            explorados.add(estadoActual)

            for action in self.problem.actions(estadoActual):
                siguienteEstado = self.problem.result(estadoActual, action)
                if siguienteEstado not in explorados and siguienteEstado not in frontera:
                    frontera.append(siguienteEstado)

        return None
    
    def manhattan_distance(self, actual, objetivo):
        return abs(actual[0] - objetivo[0]) + abs(actual[1] - objetivo[1])
    
    def euclidean_distance(self, actual, objetivo):
        return math.sqrt((actual[0] - objetivo[0]) ** 2 + (actual[1] - objetivo[1]) ** 2)

    def find_goal_state(self):
        for y in self.problem.matrix:
            for x in self.problem.matrix[y]:
                if self.problem.matrix[y][x] == (0, 255, 0):  # Color verde
                    return (x, y)
        return None

    def reconstruir_camino(self, estado_actual, mapa):
        camino = []
        while estado_actual is not None:
            camino.append(estado_actual)
            estado_actual = mapa[estado_actual[1]][estado_actual[0]]['parent']
        return camino[::-1]  # Invierte el camino para ir del inicio al final
    
    def step_cost(state, action, next_state):
            return 1

    def a_star(self):        
        estado_final = self.find_goal_state()
        estado_inicial = self.problem.initial_state()

        frontera = PriorityQueue()
        mapa = defaultdict(lambda: defaultdict(lambda: {'color': (255, 255, 255), 'G': float('inf'), 'H': 0, 'F': float('inf'), 'parent': None}))
        for y in self.problem.matrix:
            for x in self.problem.matrix[y]:
                mapa[y][x]['color'] = self.problem.matrix[y][x]

        mapa[estado_inicial[1]][estado_inicial[0]]['G'] = 0
        mapa[estado_inicial[1]][estado_inicial[0]]['H'] = self.manhattan_distance(estado_inicial, estado_final)
        #mapa[estado_inicial[1]][estado_inicial[0]]['H'] = self.euclidean_distance(estado_inicial, estado_final)

        # Tanto manhattan como euclidean distance son heurísticas admisibles para este tipo de problemas, que están en 2D, ya que ambos proporcionan una 
        # estimación del costo restante para llegar al estado final. La distancia de Manhattan es la suma de las diferencias absolutas de las coordenadas
        # de los dos puntos, mientras que la distancia euclidiana es la raíz cuadrada de la suma de los cuadrados de las diferencias de las coordenadas de
        # los dos puntos. Ambas distancias son admisibles porque nunca sobreestiman el costo restante para llegar al estado final.

        mapa[estado_inicial[1]][estado_inicial[0]]['F'] = mapa[estado_inicial[1]][estado_inicial[0]]['G'] + mapa[estado_inicial[1]][estado_inicial[0]]['H']
        frontera.put((mapa[estado_inicial[1]][estado_inicial[0]]['F'], estado_inicial))

        while not frontera.empty():
            _, estado_actual = frontera.get()
            if estado_actual == estado_final:
                print("Estado final para a_star: ", estado_actual)
                return self.reconstruir_camino(estado_actual, mapa)

            for accion in self.problem.actions(estado_actual):
                x, y = estado_actual
                siguiente_estado = self.problem.result(estado_actual, accion)
                nuevo_g = mapa[y][x]['G'] + self.problem.step_cost(estado_actual, accion, siguiente_estado)

                if nuevo_g < mapa[siguiente_estado[1]][siguiente_estado[0]]['G']:
                    mapa[siguiente_estado[1]][siguiente_estado[0]]['G'] = nuevo_g
                    mapa[siguiente_estado[1]][siguiente_estado[0]]['H'] = self.manhattan_distance(siguiente_estado, estado_final)
                    mapa[siguiente_estado[1]][siguiente_estado[0]]['F'] = mapa[siguiente_estado[1]][siguiente_estado[0]]['G'] + mapa[siguiente_estado[1]][siguiente_estado[0]]['H']
                    mapa[siguiente_estado[1]][siguiente_estado[0]]['parent'] = estado_actual
                    frontera.put((mapa[siguiente_estado[1]][siguiente_estado[0]]['F'], siguiente_estado))


    
def distanciaEuclidiana(color1, color2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(color1, color2)]))

def getSimilitud(rgb):
    # Colores de referencia
    colores_referencia = {
        "Rojo": (255, 0, 0),
        "Verde": (0, 255, 0),
        "Negro": (0, 0, 0),
        "Blanco": (255, 255, 255)
    }

    # Calcula la distancia euclidiana entre el color dado y cada color de referencia
    distancias = {nombre_color: distanciaEuclidiana(rgb, color_ref) for nombre_color, color_ref in colores_referencia.items()}

    # Devuelve el color de referencia con la menor distancia
    color_similar = min(distancias, key=distancias.get)
    color_similar = colores_referencia[color_similar]
    return color_similar

def getColores(path, reductor = 10):
    # Abrir la imagen
    imagen = Image.open(path)

    # Convertir al espacio RGB para obtener los colores de los pixeles
    img_rgb = imagen.convert('RGB')

    # Obtener las dimensiones de la imagen
    ancho, alto = imagen.size

    coloresPorPixel = defaultdict(lambda: defaultdict())

    fila = 0
    columna = 0
    for y in range(10, alto, reductor):
        for x in range(10, ancho, reductor):
            r, g, b = img_rgb.getpixel((x, y))
            rgb = (r, g, b)
            r, g, b = getSimilitud(rgb)
            coloresPorPixel[fila][columna] = (r, g, b)
            columna += 1
        columna = 0
        fila += 1
    
    return coloresPorPixel

if __name__ == "__main__":
    reductor = 4
    coloresPorPixel = getColores("Test.bmp", reductor)
    laberinto = Laberinto(coloresPorPixel)
    graphSearch = GraphSearch(laberinto)

    solucion_bfs = graphSearch.bfs()
    print("Destino en bfs: ", solucion_bfs)
    solucion_dfs = graphSearch.dfs()
    print("Destino en dfs: ", solucion_dfs)


    # Aquí ejecutas A* y guardas la solución
    solucion_a_star = graphSearch.a_star()
    #print("Solución A* encontrada:", solucion_a_star)
    
    # Dibuja el camino encontrado por A* en la imagen
    if solucion_a_star:
        explorados = set(solucion_a_star)  # Suponiendo que solucion_a_star es una lista de estados (x, y)
        graphSearch.printNewBMP(explorados, "AStarSolution.bmp", solucion_a_star[-1])
        print("Imagen con el camino de A* guardada como 'AStarSolution.bmp'")
