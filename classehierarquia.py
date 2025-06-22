from abc import ABC, abstractmethod
from typing import Any, Optional, Iterable, Iterator, Union
import math
import bisect

class DynamicArray:
    """Implementação eficiente de um array dinâmico."""
    
    def __init__(self, iterable: Optional[Iterable] = None, initial_capacity: int = 10):
        self._capacity = max(initial_capacity, 1)
        self._items = [None] * self._capacity
        self._size = 0
        
        if iterable is not None:
            for item in iterable:
                self.append(item)
    
    def __getitem__(self, index: Union[int, slice]) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        
        if not 0 <= index < self._size:
            raise IndexError("Array index out of range")
        return self._items[index]
    
    def __setitem__(self, index: int, value: Any) -> None:
        if not 0 <= index < self._size:
            raise IndexError("Array index out of range")
        self._items[index] = value
    
    def __len__(self) -> int:
        return self._size
    
    def __iter__(self) -> Iterator:
        for i in range(self._size):
            yield self._items[i]
    
    def __str__(self) -> str:
        return f"DynamicArray({list(self)})"
    
    def append(self, item: Any) -> None:
        if self._size == self._capacity:
            self._expand_capacity()
        self._items[self._size] = item
        self._size += 1
    
    def insert(self, index: int, item: Any) -> None:
        if not 0 <= index <= self._size:
            raise IndexError("Array index out of range")
        
        if self._size == self._capacity:
            self._expand_capacity()
        
        self._items[index+1:self._size+1] = self._items[index:self._size]
        self._items[index] = item
        self._size += 1
    
    def remove(self, index: int) -> Any:
        if not 0 <= index < self._size:
            raise IndexError("Array index out of range")
        
        item = self._items[index]
        self._items[index:self._size-1] = self._items[index+1:self._size]
        self._size -= 1
        return item
    
    def _expand_capacity(self) -> None:
        """Expande a capacidade de forma eficiente usando slicing."""
        self._capacity = max(1, self._capacity * 2)
        new_items = [None] * self._capacity
        new_items[:self._size] = self._items
        self._items = new_items
    
    def bubble_sort(self) -> None:
        """Ordenação otimizada com flag de verificação."""
        n = self._size
        for i in range(n):
            swapped = False
            for j in range(0, n-i-1):
                if self._items[j] > self._items[j+1]:
                    self._items[j], self._items[j+1] = self._items[j+1], self._items[j]
                    swapped = True
            if not swapped:
                break

class Matrix:
    """Classe de matriz com operações básicas."""
    
    def __init__(self, rows: int, cols: int, data: Optional[Iterable] = None):
        if rows <= 0 or cols <= 0:
            raise ValueError("Matrix dimensions must be positive")
        
        self.rows = rows
        self.cols = cols
        self.data = DynamicArray(initial_capacity=rows*cols)
        
        if data is not None:
            if len(data) != rows * cols:
                raise ValueError("Data size doesn't match matrix dimensions")
            for item in data:
                self.data.append(item)
        else:
            for _ in range(rows * cols):
                self.data.append(0)
    
    def __getitem__(self, index: tuple[int, int]) -> Any:
        row, col = index
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("Matrix index out of range")
        return self.data[row * self.cols + col]
    
    def __setitem__(self, index: tuple[int, int], value: Any) -> None:
        row, col = index
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("Matrix index out of range")
        self.data[row * self.cols + col] = value
    
    def __str__(self) -> str:
        return "\n".join(
            " ".join(str(self[i, j]) for j in range(self.cols))
            for i in range(self.rows)
        )
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions")
        
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = self[i, j] + other[i, j]
        return result
    
    def __mul__(self, other: Union['Matrix', int]) -> 'Matrix':
        if isinstance(other, int):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i, j] = self[i, j] * other
            return result
        
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Incompatible matrix dimensions")
            
            result = Matrix(self.rows, other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    result[i, j] = sum(
                        self[i, k] * other[k, j]
                        for k in range(self.cols)
                    )
            return result
        
        raise TypeError("Unsupported operand type")

class Stack:
    """Implementação eficiente de pilha."""
    
    class _Node:
        __slots__ = '_element', '_next'
        
        def __init__(self, element, next_node):
            self._element = element
            self._next = next_node
    
    def __init__(self):
        self._head = None
        self._size = 0
    
    def __len__(self) -> int:
        return self._size
    
    def is_empty(self) -> bool:
        return self._size == 0
    
    def push(self, element: Any) -> None:
        self._head = self._Node(element, self._head)
        self._size += 1
    
    def pop(self) -> Any:
        if self.is_empty():
            raise IndexError("Stack underflow")
        answer = self._head._element
        self._head = self._head._next
        self._size -= 1
        return answer
    
    def top(self) -> Any:
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._head._element

class SinglyLinkedList:
    """Lista encadeada simples com tail para eficiência."""
    
    class _Node:
        __slots__ = '_element', '_next'
        
        def __init__(self, element, next_node):
            self._element = element
            self._next = next_node
    
    def __init__(self):
        self._head = None
        self._tail = None
        self._size = 0
    
    def __len__(self) -> int:
        return self._size
    
    def is_empty(self) -> bool:
        return self._size == 0
    
    def push(self, element: Any) -> None:
        self._head = self._Node(element, self._head)
        if self.is_empty():
            self._tail = self._head
        self._size += 1
    
    def push_back(self, element: Any) -> None:
        new_node = self._Node(element, None)
        if self.is_empty():
            self._head = self._tail = new_node
        else:
            self._tail._next = new_node
            self._tail = new_node
        self._size += 1
    
    def pop(self) -> Any:
        if self.is_empty():
            raise IndexError("List is empty")
        answer = self._head._element
        self._head = self._head._next
        self._size -= 1
        if self.is_empty():
            self._tail = None
        return answer
    
    def first(self) -> Any:
        if self.is_empty():
            raise IndexError("List is empty")
        return self._head._element
    
    def last(self) -> Any:
        if self.is_empty():
            raise IndexError("List is empty")
        return self._tail._element
    
    def get_at(self, index: int) -> Any:
        if not 0 <= index < self._size:
            raise IndexError("Index out of range")
        current = self._head
        for _ in range(index):
            current = current._next
        return current._element

class Queue:
    """Fila eficiente usando SinglyLinkedList com tail."""
    
    def __init__(self):
        self._data = SinglyLinkedList()
    
    def __len__(self) -> int:
        return len(self._data)
    
    def is_empty(self) -> bool:
        return len(self._data) == 0
    
    def enqueue(self, element: Any) -> None:
        self._data.push_back(element)
    
    def dequeue(self) -> Any:
        return self._data.pop()
    
    def first(self) -> Any:
        return self._data.first()

class PriorityQueue:
    """Fila de prioridade eficiente usando lista ordenada."""
    
    def __init__(self):
        self._data = []
    
    def __len__(self) -> int:
        return len(self._data)
    
    def is_empty(self) -> bool:
        return len(self._data) == 0
    
    def insert(self, key: Any, value: Any) -> None:
        bisect.insort(self._data, (key, value))
    
    def remove_min(self) -> Any:
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return self._data.pop(0)[1]
    
    def min(self) -> Any:
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return self._data[0][1]

class Point:
    """Classe para representar um ponto 2D."""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __sub__(self, other: 'Point') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)
    
    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"

class GeometricFigure:
    """Classe para figuras geométricas com verificação de simplicidade."""
    
    def __init__(self, vertices: Iterable[Point]):
        if len(vertices) < 3:
            raise ValueError("A figure must have at least 3 vertices")
        
        self.vertices = DynamicArray()
        for vertex in vertices:
            self.vertices.append(vertex)
    
    def perimeter(self) -> float:
        if len(self.vertices) < 2:
            return 0.0
        
        total = 0.0
        n = len(self.vertices)
        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i+1)%n]
            total += (p1 - p2)
        return total
    
    def is_simple(self) -> bool:
        """Verifica se a figura é simples (sem auto-interseções)."""
        n = len(self.vertices)
        if n < 4:
            return True
        
        points = [self.vertices[i] for i in range(n)]
        
        for i in range(n):
            a1, a2 = points[i], points[(i+1)%n]
            for j in range(i+2, min(i+n-1, n)):
                b1, b2 = points[j], points[(j+1)%n]
                if self._segments_intersect(a1, a2, b1, b2):
                    return False
        return True
    
    @staticmethod
    def _segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
        """Verifica se dois segmentos se intersectam."""
        def ccw(A, B, C):
            return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)
        
        return ccw(p1,q1,q2) != ccw(p2,q1,q2) and ccw(p1,p2,q1) != ccw(p1,p2,q2)

class CafeteriaQueueSystem:
    """Sistema de fila eficiente com tracking de usuários."""
    
    def __init__(self, avg_service_time: float = 5.0):
        self.queue = Queue()
        self._user_positions = {}
        self.avg_service_time = avg_service_time
        self.current_time = 0
    
    def enqueue_user(self, user_id: str) -> None:
        self.queue.enqueue(user_id)
        self._user_positions[user_id] = len(self.queue) - 1
    
    def dequeue_user(self) -> str:
        user = self.queue.dequeue()
        self._user_positions.pop(user, None)
        self.advance_time(self.avg_service_time)
        return user
    
    def remove_user(self, user_id: str) -> bool:
        if user_id not in self._user_positions:
            return False
        
        temp = []
        found = False
        while not self.queue.is_empty():
            user = self.queue.dequeue()
            if user == user_id:
                found = True
                break
            temp.append(user)
        
        for user in temp:
            self.queue.enqueue(user)
        
        if found:
            self._user_positions.pop(user_id)
            # Atualizar posições dos usuários restantes
            self._user_positions = {user: idx for idx, user in enumerate(self.queue._data)}
        
        return found
    
    def estimated_wait_time(self, position: int) -> float:
        return position * self.avg_service_time
    
    def get_user_wait_time(self, user_id: str) -> Optional[float]:
        position = self._user_positions.get(user_id)
        return None if position is None else self.estimated_wait_time(position)
    
    def advance_time(self, minutes: float) -> None:
        self.current_time += minutes
    
    def process_next(self) -> str:
        if self.queue.is_empty():
            raise IndexError("Queue is empty")
        return self.dequeue_user()
    

# Testes para as classes implementadas 


    print("=== Teste DynamicArray ===")
arr = DynamicArray([1, 2, 3])
print(f"Array inicial: {arr}")  # Deve mostrar: DynamicArray([1, 2, 3])

arr.append(4)
print(f"Após append(4): {arr}")  # Deve mostrar: DynamicArray([1, 2, 3, 4])

arr.insert(1, 10)
print(f"Após insert(1, 10): {arr}")  # Deve mostrar: DynamicArray([1, 10, 2, 3, 4])

removido = arr.remove(2)
print(f"Removido na posição 2: {removido}")  # Deve mostrar: 2
print(f"Array após remoção: {arr}")  # Deve mostrar: DynamicArray([1, 10, 3, 4])

arr.bubble_sort()
print(f"Array ordenado: {arr}")  # Deve mostrar: DynamicArray([1, 2, 3, 4])
print("------------------------\n")