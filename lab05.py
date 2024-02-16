from PIL import Image
from collections import defaultdict
from abc import ABC, abstractmethod
from queue import Queue
from collections import deque
import heapq
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
    
    def step_cost(self, state, action, next_state):
        return 1
    
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

    def a_star(self, heuristic):
        pass

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
    for y in range(0, alto, reductor):
        for x in range(0, ancho, reductor):
            r, g, b = img_rgb.getpixel((x, y))
            rgb = (r, g, b)
            r, g, b = getSimilitud(rgb)
            coloresPorPixel[fila][columna] = (r, g, b)
            columna += 1
        columna = 0
        fila += 1
    
    return coloresPorPixel

if __name__ == "__main__":
    reductor = 15
    coloresPorPixel = getColores("Test.bmp", reductor)
    laberinto = Laberinto(coloresPorPixel)
    graphSearch = GraphSearch(laberinto)
    solucion_bfs = graphSearch.bfs()
    print(solucion_bfs)
    solucion_dfs = graphSearch.dfs()
    print(solucion_dfs)
