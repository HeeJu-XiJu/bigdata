# 최적의 코딩을 결정하는 기본 알고리즘

## 1. 가장 기본이 되는 자료구조 : 스택과 큐

### 스택 자료구조
- 먼저 들어온 데이터가 나중에 나가는 형식(선입후출)의 자료구조
- 입구와 출구가 동일한 형채로 스택을 시각화할 수 있음
![Alt text](/reference_algorithm/image.png)

예시)
삽입(5) - 삽입(2) - 삽입(3) - 삽입(7) - 삭제() - 삽입(1) - 삽입(4) - 삭제()
![Alt text](/reference_algorithm/image-1.png)

```python
stack = []
stack.append(5)
stack.append(2)
stack.append(3)
stack.append(7)
stack.pop()
stack.append(1)
stack.append(4)
stack.pop()

print(stack[::-1]) # 최상단부터 [1, 3, 2, 5]
print(stack) # [5, 2, 3, 1]
```


### 큐 자료구조
- 먼저 들어온 데이터가 먼저 나가는 형식(선입선출)의 자료구조
- 큐는 입구와 출구가 모두 뚫려있는 터널과 같은 형태로 시각화할 수 있음
![Alt text](/reference_algorithm/image-2.png)

예시)
삽입(5) - 삽입(2) - 삽입(3) - 삽입(7) - 삭제() - 삽입(1) - 삽입(4) - 삭제()
![Alt text](/reference_algorithm/image-3.png)

```python
# 시간복잡도가 작음
from collections import deque

queue = deque()
queue.append(5)
queue.append(2)
queue.append(3)
queue.append(7)
queue.popleft()
queue.append(1)
queue.append(4)
queue.popleft()

print(queue) # 먼저 들어온 순서대로 출력 deque([3, 7, 1, 4])
queue.reverse() # 역순
print(queue) # 나중에 들어온 원소부터 출력 deque([4, 1, 7, 3])
```


## 2. 우선순위에 따라 데이터를 꺼내는 자료구조

### 우선순위 큐
- 우선순위 큐는 우선순위가 가장 높은 데이터를 가장 먼저 삭제하는 자료구조
- 우선순위 큐는 데이터를 우선순위에 따라 처리하고 싶을 때 사용
![Alt text](/reference_algorithm/image-4.png)


- 우선순위 큐를 구현하는 방법
    1. 단순히 리스트 이용
    2. 힙(heap)을 이용

- 데이터의 개수가 N개일 때, 구현방식에 따라 시간복잡도가 달라짐
![Alt text](/reference_algorithm/image-5.png)


#### 힙(Heap)의 특징
- 힙은 완전 이진 트리 자료구조
- 힙에서는 항상 루트 노트를 제거
- 최소 힙
    - 루트 노트가 가장 작은 값
    - 값이 작은 데이터가 우선적으로 제거
- 최대 힙
    - 루트 노드가 가장 큰 값
    - 값이 큰 데이터가 우선적으로 제거

![Alt text](/reference_algorithm/image-6.png)


#### 완전 이진 트리
- 루트 노드부터 시작하여 왼쪽 자식 노드, 오른쪽 자식 노드 순서대로 데이터가 차례대로 삽입되는 트리
![Alt text](/reference_algorithm/image-7.png)

#### 최소 힙 구성 함수(Min-Heapify)
- (상향식) 부모 노드로 거슬러 올라가며, 부모보다 자신의 값이 더 작은 경우에 위치를 교체
- 새로운 원소가 삽입되었을 때 O(logN)의 시간 복잡도로 힙성질 유지
- 원소가 제거되었을 때 O(logN)의 시간 복잡도로 힙성질 유지
![Alt text](/reference_algorithm/image-8.png)

예시)
```python
import sys
import heapq
input = sys.stdin.readline

def heapsort(iterable):
    h = []
    result = []
    for value in iterable
    heapq.heappush(h, value)

    for i in range(len(h)):
        result.append(heapq.heappop(h))
    return result

n = int(input())
arr = []

for i in range(n):
    arr.append(int(input()))

res = heapsort(arr)

for i in range(n):
    print(res[i])
```


## 3. 트리 자료 구조
- 트리 : 가계도와 같은 계층적인 구조럴 표현
    - 루트 노드 : 부모가 없는 최상위 노드
    - 단말 노드 : 자식이 없는 노드
    - 크기 : 트리에 포함된 모든 노드의 개수
    - 깊이 : 루트 노드부터의 거리
    - 높이 : 깊이 중 최댓값
    - 차수 : 각 노드(자식방향)의 간선 개수
- 트리의 크기가 N일 때, 전체 간선의 개수는 N-1개
![Alt text](/reference_algorithm/image-9.png)

### 이진 탐색 트리
- 이진 탐색이 동작할 수 있도록 고안된 효율적인 탐색이 가능한 자료구조
- 왼쪽 자식노드 < 부모노드 < 오른쪽 자식 노드
    - 부모 노드보다 왼쪽 자식 노드가 작다.
    - 부모 노드보다 오른쪽 자식 노드가 크다.

![Alt text](/reference_algorithm/image-10.png)

#### 트리의 순회
- 트리 자료구조에 포함된 노드를 특정한 방법으로 한 번씩 방문하는 방법
    - 트리의 정보를 시각적으로 확인가능

- 전위 순회 : 루트를 먼저 방문
- 중위 순회 : 왼쪽 자식을 방문한 뒤에 루트 방문
- 후위 순회 : 오른쪽 자식을 방문한 뒤에 루트 방문
![Alt text](/reference_algorithm/image-11.png)


## 4. 바이너리 인덱스 트리
